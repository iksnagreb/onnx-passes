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
from onnx_passes.passes.util import constant_match, is_constant

# Domain used by custom operators implemented with this library
from onnx_passes.ops import DOMAIN as CUSTOM_DOMAIN
# Make custom MultiThreshold operator available to represent the fused pattern
from onnx_passes.ops.thresholds import MultiThreshold  # noqa: Used via registry
# Make custom ArgSort operator available which is used for sorting thresholds
# and threshold segments
from onnx_passes.ops.argsort import ArgSort  # noqa: Used via registry

# NumPy is used for shape calculations
import numpy as np


# Decomposes a reshape operation from shape to target shape into a squeezing,
# unsqueezing and proper reshape component
def _decompose_reshape(shape, target):
    # The target shape is allowed to contain at most one -1, in which case, the
    # value is inferred from the size of the tensor and the remaining dimensions
    if list(target).count(-1) == 1:
        # The following expression assumes target to be a list which allows
        # item assignment (might be a tuple depending on the origin of target)
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


# Applies a decomposed reshape operation to the input x which is equivalent to
# op.Reshape(x, shape) but with separate components for squeezing, unsqueezing
# and proper reshaping parts according to _decompose_reshape above.
def _apply_reshape_decomposition(op, x: ir.Value, shape: ir.Value):
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

    # Output value with applied reshape-decomposition
    return x


# Check whether the x, shape and reduction axes represent a multi-threshold
# operation. The values should already be extracted from the operator pattern,
# this check tests for the compatibility of their constant values.
def _check_multithreshold(x: ir.Value, shape: ir.Value, axes: ir.Value):
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
        return _check_multithreshold(x, shape, axes)

    def rewrite(self, op, x, shape, thresholds, weights, weighted, **kwargs):
        # Positive unit step thresholds: No weights or all weights detected to
        # be constant one
        if not weighted or constant_match(weights, 1):
            # Generate a set of unit step weight matching the thresholds
            weights = op.ConstantOfShape(
                op.Shape(thresholds), value=ir.tensor([1.0])
            )

        # Apply the reshape decomposition to the input tensor to fuse only the
        # unsqueezing part
        x = _apply_reshape_decomposition(op, x, shape)

        # Weighted, potentially non-monotonic multi-threshold function
        return op.MultiThreshold(x, thresholds, weights, _domain=CUSTOM_DOMAIN)


# Infers a fused multi-threshold function operator from the pattern according to
# the naive operator definition: y = sum(weights * (x >= thresholds))
#
# This variant of the operator fusion rule extends the match context to cover
# segments of threshold comparisons joined via a tree or Xor operators. These
# thresholding segments might the result of absorbing functions with branching
# inverses, where each branch adds another layer of segments.
@passes.verify.tolerance
@passes.register("fuse-thresholds")
class FuseThresholdSegments(Transformation, RewriteRulePass):
    @property
    def commute(self):
        return True

    def pattern(self, op, lhs, rhs, weights, axes):
        # Match the root of an optionally inverted tree of Xor operations. Well,
        # actually we want to match a single wide Xor, but ONNX Xor not n-ary...
        steps = OrValue(
            [op.Xor(lhs, rhs), op.Not(op.Xor(lhs, rhs))], tag_var="inverted"
        )

        # Type-casting turns boolean unit steps to reducible floats followed by
        # weighting for non-unit steps or non-monotonicity
        steps = op.Cast(steps, to=ir.DataType.FLOAT)
        steps = OrValue([steps, op.Mul(steps, weights)], tag_var="weighted")

        # Finally the multi-threshold output reduces over all steps removing the
        # previously expanded dimension
        return op.ReduceSum(steps, axes, keepdims=0)

    def _collect(self, value: ir.Value):
        # Terminate if the value has no producer, i.e., we reached a constant
        # input instead of a comparison
        if value.producer() is None:
            return [None]

        # If the value is produced by a GreaterOrEqual comparison, we are done
        # and can return the producer as the terminating node on this branch
        if value.producer().op_identifier() == ("", "GreaterOrEqual", ""):
            return [value.producer()]

        # If this segment branches into two more segments, continue searching
        # upwards via depth first search
        if value.producer().op_identifier() == ("", "Xor", ""):
            return [
                *self._collect(value.producer().inputs[0]),
                *self._collect(value.producer().inputs[1])
            ]

        # This branch does not seem to represent a thresholding segment,
        # indicate this by None to filter later. Terminates recursion.
        return [None]

    def _extend_context(self, lhs, rhs):
        # Depth first search along both branches to collect the thresholding
        # segment candidates
        segments = [*self._collect(lhs), *self._collect(rhs)]

        # Start with the top-level input unknown, eventually this should be the
        # same value shared by all segments
        top = None

        # All segments must terminate in a valid producer node, i.e., the
        # producer must be GreaterOrEqual with constants on the right hand side
        # and all producers must consume the same top-level input value.
        for node in segments:
            # Unsupported node or dynamic threshold candidates
            if node is None or not is_constant(node.inputs[1]):
                return segments, None, None, None

            # First segment decides on the top-level input
            if top is None:
                top = node.inputs[0]

            # Different top-level inputs for segments
            if node.inputs[0] != top:
                return segments, None, None, None

        # The top-level input must be produced by a reshape operation, similar
        # to the fusion rule above
        if (reshape := top.producer()).op_identifier() != ("", "Reshape", ""):
            return segments, top, None, None

        # Extended match context: List of comparison segments and the op-level
        # reshape input and shape
        return segments, top, reshape.inputs[0], reshape.inputs[1]

    def check(self, op, lhs, rhs, weights, axes, inverted, weighted):
        # Extend the match context to cover potential thresholding segments up
        # to the top-level input and reshaping
        segments, _, x, shape = self._extend_context(lhs, rhs)

        # Could not extend the context to a shared top-level input with reshape
        if x is None or shape is None:
            return False

        # Same match condition as the threshold fusion rule to check whether the
        # top and the already pattern-matched tail are valid.
        return _check_multithreshold(x, shape, axes)

    def rewrite(self, op, lhs, rhs, weights, axes, inverted, weighted):
        # Extend the match context to cover potential thresholding segments up
        # to the top-level input and reshaping
        segments, top, x, shape = self._extend_context(lhs, rhs)

        # Number of thresholds/steps per segment, must be the same for all
        # segments to allow concatenation
        num_thresholds = op.Cast(
            op.Shape(segments[0].inputs[1], start=-1), to=ir.DataType.FLOAT
        )

        # Assume implicit positive unit-step weight for all steps if no explicit
        # weighting is present
        if not weighted:
            weights = op.Constant(value_float=1.0)

        # If the output of the Xor is inverted, negate the weights to invert
        # all step directions
        if inverted:
            weights = op.Neg(weights)

        # Replicate the initial weights for each threshold segment
        weights = len(segments) * [weights]

        # Start collecting threshold parameter segments as list of values
        thresholds = []

        # Collect threshold tensors from all segments into a list of values to
        # be concatenated
        for node in segments:
            # Prepare an extra trailing axis into which the segments will be
            # concatenated (we need to sort segments before flattening)
            thresholds.append(
                op.Unsqueeze(node.inputs[1], op.Constant(value_ints=[-1]))
            )

        # Expand all threshold and weight tensors to account for broadcasting
        for index, (ts, ws) in enumerate(zip(thresholds, weights)):
            thresholds[index] = op.Expand(ts, op.Shape(op.Max(*thresholds)))
            weights[index] = op.Expand(ws, op.Shape(op.Max(*thresholds)))

        # Concatenate thresholds and weights and sort the threshold segments in
        # ascending order
        thresholds = op.Concat(*thresholds, axis=-1)
        weights = op.Concat(*weights, axis=-1)

        indices = op.ArgSort(thresholds, axis=-1, _domain=CUSTOM_DOMAIN)

        thresholds = op.GatherElements(thresholds, indices, axis=-1)
        weights = op.GatherElements(weights, indices, axis=-1)

        # Generate alternating pattern of +1 and -1 weights to set the step
        # direction per segment
        directions = op.Constant(
            value_floats=[(-1.0) ** i for i in range(len(segments))]
        )

        weights = op.Mul(directions, weights)

        # Flatten the final two dimensions to concatenate all segments into the
        # threshold axis
        thresholds = op.Reshape(
            thresholds,
            op.Concat(
                op.Shape(thresholds, end=-2),
                op.Constant(value_ints=[-1]),
                axis=0
            )
        )

        weights = op.Reshape(
            weights,
            op.Concat(
                op.Shape(weights, end=-2),
                op.Constant(value_ints=[-1]),
                axis=0
            )
        )

        # Sort the flattened thresholds and corresponding weights into ascending
        # order (globally, not just per segment)
        indices = op.ArgSort(thresholds, axis=-1, _domain=CUSTOM_DOMAIN)

        thresholds = op.GatherElements(thresholds, indices, axis=-1)
        weights = op.GatherElements(weights, indices, axis=-1)

        # If the Xor-tree has inverted outputs, the logic is as follows: If the
        # input exceeds none of the thresholds, that is zero steps, but with Not
        # inverting the outputs, that is all the steps. As inverted outputs also
        # negates the weights, this biases the output to -number of steps, which
        # we counteract by biasing in the opposite direction.
        bias = op.Squeeze(
            num_thresholds if inverted else op.Constant(value_float=0.0)
        )

        # Apply the reshape decomposition to the input tensor to fuse only the
        # unsqueezing part
        x = _apply_reshape_decomposition(op, x, shape)

        # Weighted, potentially non-monotonic multi-threshold function with all
        # combined thresholds and weights.
        return op.Add(
            op.MultiThreshold(x, thresholds, weights, _domain=CUSTOM_DOMAIN),
            bias
        )
