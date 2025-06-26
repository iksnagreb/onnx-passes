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


# Infers a fused multi-threshold function operator from the pattern according to
# the naive operator definition: y = sum(weights * (x >= thresholds))
@passes.verify.tolerance
@passes.register("fuse-thresholds")
class FuseThresholds(Transformation, RewriteRulePass):
    @property
    def commute(self):
        return True

    def pattern(self, op, x, thresholds, weights, shape, axes):
        # Comparison of inputs and all corresponding thresholds: Expand input
        # dimensions to match the threshold parameter shape via broadcasting
        steps = op.LessOrEqual(thresholds, op.Reshape(x, shape, allowzero=0))

        # Type-casing turns boolean unit steps to reducible floats followed by
        # weighting for non-unit steps or non-monotonicity
        steps = OrValue([steps, op.Cast(steps)], tag_var="cast")
        steps = OrValue([steps, op.Mul(steps, weights)], tag_var="weighted")

        # Finally the multi-threshold output reduces over all steps removing the
        # previously expanded dimension
        return op.ReduceSum(steps, axes, keepdims=0)

    def check(self, op, x, shape, axes, **kwargs):
        # The expansion shape must be constant and match the input except for
        # the expanded dimension, which must be 1
        if not constant_match(shape, [*x.shape, 1]):
            return False

        # The sum-reduction must operate on this expanded final axis and
        # no other axes
        if not constant_match(axes, -1):
            return False

        # TODO: It is assumed that the broadcasting of the thresholds, weights
        #  and input tensor shapes is valid and there is no need to check here

        # All checks passed - pattern matched and may be rewritten
        return True

    def rewrite(self, op, x, thresholds, weights, weighted, **kwargs):
        # Positive unit step thresholds: No weights or all weights detected to
        # be constant one
        # if not weighted or constant_match(weights, 1):
        #     return op.MultiThreshold(x, thresholds, _domain=custom)
        # Weighted, potentially non-monotonic multi-threshold function
        return op.MultiThreshold(x, thresholds, weights, _domain=CUSTOM_DOMAIN)
