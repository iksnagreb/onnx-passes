# Streamlining passes are Transformations with the frontend composed of other
# passes
from onnx_passes.passes.base import Transformation, RewriteRuleSetPass
from onnx_passes.passes.compose import ComposePass

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# Threshold fusion and inlining passes to make the MultiThreshold custom-op
# streamlinable
import onnx_passes.passes.inline.thresholds  # noqa: Used via key
import onnx_passes.passes.fusion.thresholds  # noqa: Used via key

# Range annotations allow the conversion of rounding operations into thresholds
# representations
import onnx_passes.passes.annotation.range  # noqa: Used via key

# Make default streamlining passes registered as "streamline" available
import onnx_passes.passes.streamline  # noqa: Used via key


# Exhaustive threshold streamlining transformations: Cleans up the graph, infers
# threshold representations from quantizers, rounding, etc. based on range
# annotations and inlines thresholds into the graph such that the comparisons
# serve as sinks for streamlining elementwise operations.
@passes.verify.tolerance
class _StreamlineThresholds(ComposePass, Transformation):
    # Ordered sequence of passes and pass collections to be applied for each
    # iteration of streamlining
    __passes__ = [
        # Range annotation needs a cleaned-up graph with proper shape
        # annotations, attributes and constants lifted to initializers, etc.
        "shape-inference",
        "fold-constants",
        "cleanup",
        # Add range information to all IR values to enable converting rounding
        # operations (Round, Ceil, Floor) to MultiThresholds.
        "range-annotation",
        # Infer threshold representations from rounding to integers and
        # immediately inline the representation for further streamlining
        "infer-thresholds",
        "inline-thresholds",
        # Conclude with usual streamlining and streamlining of shapes - these
        # are exhaustive and might enable more threshold inference
        "streamline"
    ]

    # Keep iterating the streamlining passes until the model stops changing
    __exhaustive__ = True


# The non-exhaustive frontend pass for streamlining thresholds: Adds a
# non-repeating threshold fusion step to the exhaustive sequence above
@passes.verify.tolerance
@passes.register("streamline-thresholds")
class StreamlineThresholds(ComposePass, Transformation):
    # Ordered sequence of passes and pass collections to be applied for each
    # iteration of streamlining
    __passes__ = [
        # Exhaustive streamlining of threshold operations operating on the
        # inlined representation of MultiThreshold custom-ops
        _StreamlineThresholds,
        # Convert back to MultiThreshold custom-ops representing the
        # thresholding operations
        "fuse-thresholds",
        # After rearranging the graph, make sure everything is properly
        # annotated with shapes
        "shape-inference",
        # After grouping constants, more of them should be foldable into
        # Constant operators
        "fold-constants",
        # Finally cleanup the graph by removing nodes, constants, attributes,
        # etc. no longer needed and bringing it into an ordered state
        "cleanup"
    ]

    # Iterating this exhaustively will end up in endless cycles
    __exhaustive__ = False


# ir.Model, ir.DataType, ir.passes.PassResult, ir.from_proto, ir.to_proto, ...
import onnx_ir as ir

# Collects range annotations from IR Value with reasonable fallbacks
from onnx_passes.passes.annotation.range import _get_range  # noqa: Protected
# Alias for expanding a tensor of ones to the same shape and type as another
# tensor
from onnx_passes.passes.util import ones_like
# Custom ONNX domain providing a reference implementation of MultiThreshold
from onnx_passes.ops import DOMAIN as CUSTOM_DOMAIN

# Function with partially applied arguments: Used to greate a generator
# mechanism inserting operators into a pattern template
from functools import partial

# Working with constant tensors/values, and inserting constants such as infinity
import numpy as np


# Infers threshold representations from range-annotated Round operators to
# complete the conversion of quantizers to MultiThreshold operators
@passes.verify.tolerance
@passes.register("infer-thresholds")
class ConvertRoundToThresholds(Transformation, RewriteRuleSetPass):
    # Initializes the maximum threshold conversion threshold to prevent
    # generating exessive amounts of steps
    def __init__(
            self, config: dict | None, state: dict | None, max_thresholds=255
    ):
        super().__init__(config, state)
        # Default threshold conversion threshold
        self._max_thresholds = max_thresholds

    @property
    def max_thresholds(self):
        # Note: As all passes share the configuration, this could be changed
        return (self.config.setdefault("infer_thresholds", {})
                .setdefault("max", self._max_thresholds))

    def pattern(self):
        return [
            lambda op, x: op.Round(x),
            lambda op, x: op.Ceil(x),
            lambda op, x: op.Floor(x),
        ]

    def check(self):
        def _check(op, x):
            # Collect the range annotation on the input value
            _min, _max = _get_range(x)
            # Upper and lower bound must be present and constrain the range to
            # less the maximum number of thresholds allowed for conversion
            if _min is not None and _max is not None:
                if np.ceil(_max - _min) <= self.max_thresholds:
                    # Must have shape annotations
                    return x.shape is not None
            # No range annotation restricting the thresholds to generate
            return False

        # Same match condition for all rounding functions
        # TODO: Well, we might get +/- 1 extra threshold by not properly
        #  considering the rounding mode, but this is probably good enough...
        return [_check, _check, _check]

    def rewrite(self):
        def _rewrite(mode, op, x):
            # Collect the range annotation on the input value
            _min, _max = _get_range(x)

            # Threshold generation depends on the rounding mode
            thresholds = {
                # TODO: Strictly, Round rounds to even, so the steps should
                #  alternate between >= and > comparisons or equivalently adding
                #  a small epsilon turning > into >=.
                "Round": np.arange(np.round(_min), np.round(_max), 1) + 0.5,
                "Ceil": np.arange(np.ceil(_min), np.ceil(_max), 1),
                "Floor": np.arange(np.floor(_min), np.floor(_max), 1) + 1.0,
            }[mode]

            # Unsqueeze matching input dimensions from the threshold tensor
            # TODO: Is this really necessary? Broadcasting should be valid
            #  anyway...?
            thresholds = np.reshape(thresholds, (*(1 for _ in x.shape), -1))

            # Pack numpy array as ONNX operator
            thresholds = op.CastLike(
                op.Constant(value=ir.tensor(thresholds)), x
            )

            # Rounding functions are monotonic unit steps
            weights = ones_like(op, thresholds)

            # Compose replacement pattern - same for all rounding modes
            return op.Add(
                # Multi threshold representation of rounding as unit steps
                op.MultiThreshold(
                    x, thresholds, weights, _domain=CUSTOM_DOMAIN
                ),
                # Bias accounting for the minimum of the range
                op.CastLike(op.Constant(value=ir.tensor(_min)), x)
            )

        # Instantiate the pattern template for all three rounding modes
        return [
            partial(_rewrite, "Round"),
            partial(_rewrite, "Ceil"),
            partial(_rewrite, "Floor"),
        ]
