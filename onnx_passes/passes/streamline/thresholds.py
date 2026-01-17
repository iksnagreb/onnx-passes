# Streamlining passes are Transformations with the frontend composed of other
# passes
from onnx_passes.passes.base import (
    Transformation, RewriteRuleSetPass, RewriteRulePass
)
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
        # Bring all thresholds into ascending order as the normal form required
        # for efficient implementation via binary search
        "sort-thresholds",
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


def _sort_thresholds(thresholds: np.ndarray, weights: np.ndarray):
    # Broadcast threshold and step direction weights to make indices compatible
    thresholds, weights = np.broadcast_arrays(thresholds, weights)

    # Sort thresholds and step direction weights in ascending order along the
    # last axis
    return (
        np.take_along_axis(thresholds, np.argsort(thresholds), axis=-1),
        np.take_along_axis(weights, np.argsort(thresholds), axis=-1),
    )


@passes.verify.tolerance
@passes.register("sort-thresholds")
class SortThresholds(Transformation, RewriteRulePass):
    def pattern(self, op, x, thresholds, weights):
        return op.MultiThreshold(x, thresholds, weights, _domain=CUSTOM_DOMAIN)

    def check(self, op, x, thresholds, weights):
        # Thresholds must be constant to be decomposed, otherwise there are no
        # known values to extract and iterate
        if (thresholds := ir.convenience.get_const_tensor(thresholds)) is None:
            return False

        # Weights must be constant to be decomposed, otherwise there are no
        # known values to extract and iterate
        if (weights := ir.convenience.get_const_tensor(weights)) is None:
            return False

        # Convert ONNX tensor to NumPy array for further checks
        weights, thresholds = weights.numpy(), thresholds.numpy()

        # If there is only a single threshold, there is nothing to sort
        if len(thresholds.shape) == 0 or thresholds.shape[-1] == 1:
            return False

        # Do not sort again if thresholds are already sorted
        return np.any(np.sort(thresholds, axis=-1) != thresholds)

    def rewrite(self, op, x, thresholds, weights):
        # Sort thresholds and associated step direction weights in numpy format
        thresholds, weights = _sort_thresholds(
            ir.convenience.get_const_tensor(thresholds).numpy(),
            ir.convenience.get_const_tensor(weights).numpy(),
        )

        # Try to unbroadcast the sorted parameters (while sorting these might be
        # broadcast to ensure compatible indices)
        thresholds = unbroadcast(thresholds)
        weights = unbroadcast(weights)

        # Insert thresholds back into ONNX constants and make sure the type is
        # the same as the input as required by ONNX standard of elementwise
        thresholds = op.Constant(value=ir.tensor(thresholds))
        weights = op.Constant(value=ir.tensor(weights))

        # Replacement pattern: MultiThreshold operator with thresholds and
        # weights sorted into ascending order of thresholds
        return op.MultiThreshold(x, thresholds, weights, _domain=CUSTOM_DOMAIN)


@passes.verify.tolerance
@passes.register("unbroadcast")
class UnbroadcastThresholds(Transformation, RewriteRulePass):
    def pattern(self, op, x, thresholds, weights):
        return op.MultiThreshold(x, thresholds, weights, _domain=CUSTOM_DOMAIN)

    def check(self, op, x, thresholds, weights):
        # Thresholds must be constant to be decomposed, otherwise there are no
        # known values to extract and iterate
        if (thresholds := ir.convenience.get_const_tensor(thresholds)) is None:
            return False

        # Weights must be constant to be decomposed, otherwise there are no
        # known values to extract and iterate
        if (weights := ir.convenience.get_const_tensor(weights)) is None:
            return False

        # Check whether the threshold tensor can be unbroadcast
        if len(unbroadcast(thresholds.numpy()).shape) < len(thresholds.shape):
            return True

        # Check whether the weights tensor can be unbroadcast
        if len(unbroadcast(weights.numpy()).shape) < len(weights.shape):
            return True

        # Do not unbroadcast again if thresholds are already unbroadcast
        return False

    def rewrite(self, op, x, thresholds, weights):
        # Unbroadcast thresholds and step direction weights in numpy format
        thresholds = ir.convenience.get_const_tensor(thresholds).numpy()
        weights = ir.convenience.get_const_tensor(weights).numpy()

        # Insert thresholds back into ONNX constants and make sure the type is
        # the same as the input as required by ONNX standard of elementwise
        thresholds = op.Constant(value=ir.tensor(unbroadcast(thresholds)))
        weights = op.Constant(value=ir.tensor(unbroadcast(weights)))

        # Replacement pattern: MultiThreshold operator with thresholds and
        # weights unbroadcast if possible
        return op.MultiThreshold(x, thresholds, weights, _domain=CUSTOM_DOMAIN)


def _decompose_monotonicity(thresholds: np.ndarray, weights: np.ndarray):
    # Get rid of all zero steps by setting the corresponding threshold to
    # infinity, i.e., x > inf = False
    thresholds = np.where(weights == 0.0, np.inf, thresholds)

    # Decreasing (< 0.0) and increasing (> 0.0) segments of the thresholds, no
    # need to return the weights as these are always abs(weights)
    return (
        np.where(weights < 0.0, thresholds, np.inf),
        np.where(weights > 0.0, thresholds, np.inf)
    )


# Decomposes non-monotonic (i.e., increasing and decreasing) thresholds into the
# difference of two increasing multi-threshold functions
@passes.verify.tolerance
@passes.register("decompose-thresholds")
class ThresholdMonotonicityDecomposition(Transformation, RewriteRulePass):
    def pattern(self, op, x, thresholds, weights):
        return op.MultiThreshold(x, thresholds, weights, _domain=CUSTOM_DOMAIN)

    def check(self, op, x, thresholds, weights):
        # Thresholds must be constant to be decomposed, otherwise there are no
        # known values to extract and iterate
        if (thresholds := ir.convenience.get_const_tensor(thresholds)) is None:
            return False

        # Weights must be constant to be decomposed, otherwise there are no
        # known values to extract and iterate
        if (weights := ir.convenience.get_const_tensor(weights)) is None:
            return False

        # Decompose if there are increasing (positive) and decreasing (negative
        # weights)
        return np.any(weights.numpy() <= 0)

    def rewrite(self, op, x, thresholds, weights):
        # We start with the initial thresholds and weights in numpy format
        _thresholds = ir.convenience.get_const_tensor(thresholds).numpy()
        _weights = ir.convenience.get_const_tensor(weights).numpy()

        # If all weights are negative, this is a monotonically decreasing
        # function, we can simply flip all step directions and negate the output
        if np.all(_weights <= 0):
            return op.Neg(
                op.MultiThreshold(
                    x, thresholds, op.Abs(weights), _domain=CUSTOM_DOMAIN
                )
            )

        # Decompose the threshold tensor into increasing and decreasing segments
        decreasing, increasing = _decompose_monotonicity(_thresholds, _weights)

        # Insert thresholds back into ONNX constants and make sure the type is
        # the same as the input as required by ONNX standard of elementwise
        increasing = op.CastLike(op.Constant(value=ir.tensor(increasing)), x)
        decreasing = op.CastLike(op.Constant(value=ir.tensor(decreasing)), x)

        # Replace weights by positive steps with the same magnitude (opposite
        # direction steps are skipped by mapping to infinity)
        weights = op.Abs(weights)

        # Replacement pattern: MultiThreshold decomposed into a difference of
        # two monotonically increasing MultiThreshold functions
        return op.Sub(
            op.MultiThreshold(x, increasing, weights, _domain=CUSTOM_DOMAIN),
            op.MultiThreshold(x, decreasing, weights, _domain=CUSTOM_DOMAIN)
        )


def _decompose_granularity_torch(thresholds):
    # Try to dynamically load PyTorch and raise an exception if it is not
    # available
    try:
        import torch
    except ModuleNotFoundError:
        raise RuntimeError(
            "PyTorch is required for threshold granularity decomposition"
        )

    # Split the shape into the fine-granular bias component and the per-channel
    # threshold component (both cover the channel axis, i.e., -2)
    b_shape, t_shape = thresholds.shape[:-1], thresholds.shape[-2:]

    # Convert from NumPy to PyTorch format for accelerated minimization
    thresholds = torch.as_tensor(thresholds, dtype=torch.float32)

    # Initial bias and per-channel thresholds parameter tensor to be optimized
    bias = torch.randn(b_shape, requires_grad=True)
    t = torch.randn(t_shape, requires_grad=True)

    # Set up an LBFGS optimizer to solve for the bias as per-channel threshold
    # which reconstruct the fine-granular thresholds via broadcasting
    optimizer = torch.optim.LBFGS([bias, t], line_search_fn="strong_wolfe")

    # Do not consider infinity thresholds for optimization: There is no bias
    # which could pull these away from infinity into any direction
    infinity = torch.abs(thresholds) >= torch.inf

    # Wraps objective function evaluation and gradient calculation as required
    # by the LBFGS step
    def closure():
        # Clear the gradients from lat iteration
        optimizer.zero_grad()
        # Optimization objective: Reconstruction error for recovering the fine
        # granular thresholds as thresholds = t - bias
        loss = torch.sum(
            torch.square(
                torch.where(
                    infinity, 0, (bias.unsqueeze(-1) - (t - thresholds))
                )
            )
        )
        # Compute the gradient of the objective with respect to t and the bias
        loss.backward()
        # LBFGS needs the loss
        return loss

    # Optimize the reconstruction objective via LBFGS optimizer step
    optimizer.step(closure)

    # Convert the result back to NumPy format (force to detach from the
    # computational graph)
    return unbroadcast(t.numpy(force=True)), bias.numpy(force=True)


# Threshold granularity decomposition involves approximate unbroadcasting of
# replacement thresholds at ist core
from onnx_passes.passes.util import unbroadcast


def _decompose_granularity_naive(thresholds: np.ndarray):
    # Draw some random thresholds covering only the final axis to replace
    # the fine-granular thresholds. Thresholds should always be sorted.
    t = np.sort(np.random.randn(*thresholds.shape[-2:]), axis=-1)
    # Generate the corresponding bias in front of the thresholds acting
    # elementwise on all axes except for the internal threshold axis.
    bias = np.round(t - thresholds)[..., :1]
    # As the original derivation does not assume rounding the bias, add this
    # correction term before unbroadcasting while allowing small deviations.
    return unbroadcast(thresholds + bias, approximate=True), bias[..., 0]


def _decompose_granularity(thresholds: np.ndarray):
    # First try the naive decomposition via unbroadcasting and with bias rounded
    # to integers: Successful if this yields a tensor of rank two or less
    if len((naive := _decompose_granularity_naive(thresholds))[0].shape) <= 2:
        return naive

    # If the naive solution is not successful, we need to solve an optimization
    # problem to approximate theta = theta' - bias
    return _decompose_granularity_torch(thresholds)


# Decomposes fine-granular (e.g., per-element) thresholds into a fine-granular
# integer-valued elementwise addition and per-channel thresholds if possible.
@passes.verify.tolerance
@passes.register("decompose-thresholds")
class ThresholdGranularityDecomposition(Transformation, RewriteRulePass):
    def pattern(self, op, x, thresholds, weights):
        return op.MultiThreshold(x, thresholds, weights, _domain=CUSTOM_DOMAIN)

    def check(self, op, x, thresholds, weights):
        # Thresholds must be constant to be decomposed, otherwise there are no
        # known values to extract and iterate
        if (thresholds := ir.convenience.get_const_tensor(thresholds)) is None:
            return False

        # If the threshold tensor already is of at most per-channel granularity,
        # there is no need to decompose
        if len(unbroadcast(thresholds.numpy()).shape) <= 2:
            return False

        # Decompose the threshold tensor into new per-channel thresholds and a
        # corresponding per-element bias
        t, bias = _decompose_granularity(thresholds.numpy())

        # The threshold decomposition must actually be at most per-channel
        # granularity. No need to verify the bias shape, this is allowed to be a
        # fine-granular tensor.
        if len(t.shape) > 2:
            return False

        # Accept this decomposition, even if it might sometimes introduce a
        # small drop in model quality/accuracy
        return True

    def rewrite(self, op, x, thresholds, weights):
        # Decompose the threshold tensor into new per-channel thresholds and a
        # corresponding per-element bias
        thresholds, bias = _decompose_granularity(
            ir.convenience.get_const_tensor(thresholds).numpy()
        )

        # If the step weights are constant, try unbroadcasting as an unrelated
        # optimization along the way
        if (w := ir.convenience.get_const_tensor(weights)) is not None:
            weights = op.Constant(value=ir.tensor(unbroadcast(w.numpy())))

        # Insert thresholds back into ONNX constants and make sure the type is
        # the same as the input as required by ONNX standard of elementwise
        thresholds = op.CastLike(op.Constant(value=ir.tensor(thresholds)), x)

        # Insert the new bias into ONNX constants and make sure the type is the
        # same as the input as required by the ONNX standard of elementwise
        bias = op.CastLike(op.Constant(value=ir.tensor(bias)), x)

        # Replacement pattern: MultiThreshold decomposed into additive bias and
        # per-channel thresholds
        return op.MultiThreshold(
            op.Add(x, bias), thresholds, weights, _domain=CUSTOM_DOMAIN
        )


def _decompose_multiplicity(thresholds: np.ndarray, weights: np.ndarray):
    # Remember original shape of leading dimensions before flattening the arrays
    # to simplify the padding and iteration below
    thresholds_shape = thresholds.shape[:-1]
    weights_shape = weights.shape[:-1]

    thresholds = thresholds.reshape((-1, thresholds.shape[-1]))
    weights = weights.reshape((-1, weights.shape[-1]))

    # Get rid of all "dead" thresholds by setting their multiplicity to zero
    weights = np.where(thresholds >= np.inf, 0, weights)
    # np.where broadcasts the shapes
    weights_shape = np.broadcast_shapes(thresholds_shape, weights_shape)

    # Number of steps per set of thresholds and maximum number of steps of the
    # entire set
    n = np.sum(np.abs(weights), axis=-1, keepdims=True)
    max_n = np.max(np.sum(np.abs(weights), axis=-1, keepdims=True))

    # Add infinity padding to the threshold list, pad each set of thresholds by
    # the amount necessary to have the same number of thresholds for each set
    thresholds = np.pad(thresholds, ((0, 0), (0, 1)), constant_values=np.inf)
    weights = np.concatenate((weights, max_n - n), axis=-1)

    # Number of repetitions of each threshold, i.e., integer-valued threshold
    # multiplicity
    repeats = np.abs(weights).astype(np.int64)

    # Repeat all thresholds according to their multiplicity, i.e., magnitude of
    # the associated step direction weight and map weights to +/- unit steps
    thresholds = [np.repeat(ts, num) for ts, num in zip(thresholds, repeats)]
    weights = [np.repeat(np.sign(ws), num) for ws, num in zip(weights, repeats)]

    # Restore original shape of the leading dimensions of the threshold and
    # weight tensor
    return (
        np.reshape(np.asarray(thresholds), (*thresholds_shape, -1)),
        np.reshape(np.asarray(weights), (*weights_shape, -1))
    )


# Decomposes non-unit step thresholds by replicating steps according to their
# multiplicity if possible (all weights are integers).
@passes.verify.tolerance
@passes.register("decompose-thresholds")
class ThresholdMultiplicityDecomposition(Transformation, RewriteRulePass):
    def pattern(self, op, x, thresholds, weights):
        return op.MultiThreshold(x, thresholds, weights, _domain=CUSTOM_DOMAIN)

    def check(self, op, x, thresholds, weights):
        # Thresholds must be constant to be decomposed, otherwise there are no
        # known values to extract and iterate
        if (thresholds := ir.convenience.get_const_tensor(thresholds)) is None:
            return False

        # Weights must be constant to be decomposed, otherwise there are no
        # known values to extract and iterate
        if (weights := ir.convenience.get_const_tensor(weights)) is None:
            return False

        # Convert ONNX tensor to NumPy array for further checks
        weights, thresholds = weights.numpy(), thresholds.numpy()

        # All weights must be integers to apply the decomposition, as we cannot
        # fractionally replicate a step
        if np.any(np.asarray(np.round(weights), dtype=np.int64) != weights):
            return False

        # Decompose if there are weights that are not +/- 1 or if for all axes
        # there are infinity thresholds which can be removed
        return (np.any(np.abs(weights) != 1.0)
                or np.all(np.any(thresholds >= np.inf, axis=-1)))

    def rewrite(self, op, x, thresholds, weights):
        # Apply the decomposition to thresholds and weights converted to NumPy
        # format
        thresholds, weights = _decompose_multiplicity(
            ir.convenience.get_const_tensor(thresholds).numpy(),
            ir.convenience.get_const_tensor(weights).numpy()
        )

        # Insert thresholds back into ONNX constants and make sure the type is
        # the same as the input as required by ONNX standard of elementwise
        thresholds = op.Constant(value=ir.tensor(thresholds))
        weights = op.Constant(value=ir.tensor(weights))

        # Replacement pattern: MultiThreshold operator with thresholds and
        # weights repeated according to their multiplicity
        return op.MultiThreshold(x, thresholds, weights, _domain=CUSTOM_DOMAIN)
