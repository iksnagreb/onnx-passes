# ir.Model, ir.passes.PassResult, ir.from_proto, ir.to_proto, ...
import onnx_ir as ir

# Matching against one value pattern from a selection of alternative patterns
from onnxscript.rewriter.pattern import OrValue

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# All threshold transformations are transformations derived from pattern-based
# rewrite rules
from onnx_passes.passes.base import Transformation, RewriteRulePass
# Checking ir.Value for being constants and comparing constants to be identical
from onnx_passes.passes.util import constant_match

# Domain used by custom operators implemented with this library
from onnx_passes.ops import DOMAIN as CUSTOM_DOMAIN
# Make custom MultiThreshold operator available to represent the fused pattern
from onnx_passes.ops.thresholds import MultiThreshold  # noqa: Used indirectly

# NumPy is used for shape calculations
import numpy as np


# Decomposes a reshape operation from shape to target shape into a squeezing,
# unsqueezing and proper reshape component
def _decompose_reshape(shape, target):
    # The target shape is allowed to contain at most one -1, in which case, the
    # value is inferred from the size of the tensor and the remaining dimensions
    if list(target).count(-1) == 1:
        # # The following expression assumes target to be a list which allows
        # item assignment (might be a tuple depending on the origin of target)s
        target = list(target)
        # Replace the inferred dimension by the number of elements missing from
        # the target dimension compared to the input dimension
        target[target.index(-1)] = int(np.prod(shape) / np.abs(np.prod(target)))

    # The squeeze component are all axes which can be squeezed from the input
    squeeze = [int(axis) for axis, n in enumerate(shape) if n == 1]

    # The reshaping component now covers all non-empty target dimensions, i.e.,
    # those which can not be squeezed from the target
    reshape = [int(n) for axis, n in enumerate(target) if n != 1]

    # The unsqueeze component are all axes which must be restored for the target
    # shape, i.e., those which can be squeezed from the target
    unsqueeze = [int(axis) for axis, n in enumerate(target) if n == 1]

    # Tuple describing the reshape decomposition in the order in which it should
    # be applied
    return squeeze, reshape, unsqueeze


# Infers a fused multi-threshold function operator from the pattern according to
# the naive operator definition: y = sum(weights * (x >= thresholds))
@passes.verify.tolerance
@passes.register("fuse-thresholds")
class FuseThresholds(Transformation, RewriteRulePass):
    @property
    def commute(self):
        return True

    def pattern(self, op, x, thresholds, weights, shape, axes, allowzero):
        # Comparison of inputs and all corresponding thresholds: Expand input
        # dimensions to match the threshold parameter shape via broadcasting
        steps = op.GreaterOrEqual(
            op.Reshape(x, shape, allowzero=allowzero), thresholds
        )

        # Type-casting turns boolean unit steps to reducible floats followed by
        # weighting for non-unit steps or non-monotonicity
        steps = OrValue([steps, op.Cast(steps)], tag_var="cast")
        steps = OrValue([steps, op.Mul(steps, weights)], tag_var="weighted")

        # Finally the multi-threshold output reduces over all steps removing the
        # previously expanded dimension
        return op.ReduceSum(steps, axes, keepdims=0)

    def check(self, op, x, shape, axes, allowzero, **kwargs):
        # The expansion shape must be constant to check and ensure compatibility
        # between the shape and the thresholds shape
        if (shape := ir.convenience.get_const_tensor(shape)) is None:
            return False

        # The input shape must be available and static to check for shape
        # compatibility
        if x.shape is None or not x.shape.is_static():
            return False

        # Decompose the reshape in front of the pattern: Due to fusion of
        # successive reshape operations, this might not look like the unsqueeze
        # operation required to match the number of thresholds
        _, _, unsqueeze = _decompose_reshape(x.shape, shape.numpy())

        # Check whether there is an unsqueezing of the final dimension as a
        # component of the decomposed reshape
        if (len(shape.numpy()) - 1) not in unsqueeze:
            return False

        # To avoid issues with correct interpretation of the allowzero attribute
        # and also as empty tensors probably do not make much sense here, reject
        # any instances where there are zeros in the target shape
        if np.any(shape.numpy() == 0):
            return False

        # The reduction must operate on the expanded final axis only, this means
        # (1), the axis must be a constant, (2), the axis must be a single value
        # and, (3), the axis must be either -1 or the last axis
        if (axes := ir.convenience.get_const_tensor(axes)) is not None:
            if axes.numpy().size == 1:
                if axes.numpy().item() in {-1, len(shape.numpy()) - 1}:
                    return True

        # Last set of nested checks on the reduction axis did not pass, this is
        # something else, not thresholds
        return False

    def rewrite(self, op, x, shape, thresholds, weights, weighted, **kwargs):
        # The target shape is known to be a constant tensor, can safely be
        # converted to numpy
        shape = ir.convenience.get_const_tensor(shape).numpy()
        # Reshape decomposition to restore any non-matched Reshape in front of
        # the thresholding, only the final unsqueeze can be fused
        squeeze, reshape, unsqueeze = _decompose_reshape(x.shape, shape)

        # Remove the final unsqueezed dimension which is accounted for by the
        # fused operator
        unsqueeze.remove(len(shape) - 1)

        # If the reshape has a squeezing component, insert an explicit squeeze
        # operator at the input
        if squeeze:
            x = op.Squeeze(x, op.Constant(value_ints=squeeze))

        # Proper reshape component, should always be present but might be empty
        # for scalars or effectively scalar tensors
        x = op.Reshape(x, op.Constant(value_ints=reshape))

        # If the reshape has an unsqueezing component, insert an explicit
        # unsqueeze operator at the input
        if unsqueeze:
            x = op.Unsqueeze(x, op.Constant(value_ints=unsqueeze))

        # Positive unit step thresholds: No weights or all weights detected to
        # be constant one
        if not weighted or constant_match(weights, 1):
            # Generate a set of unit step weight matching the thresholds
            weights = op.ConstantOfShape(
                op.Shape(thresholds), value=ir.tensor([1.0])
            )

        # Weighted, potentially non-monotonic multi-threshold function
        return op.MultiThreshold(x, thresholds, weights, _domain=CUSTOM_DOMAIN)
