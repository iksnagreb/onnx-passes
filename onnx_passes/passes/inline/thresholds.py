# ir.Value
import onnx_ir as ir

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# All threshold transformations are transformations derived from pattern-based
# rewrite rules
from onnx_passes.passes.base import Transformation, RewriteRulePass
# Domain used by custom operators implemented with this library
from onnx_passes.ops import DOMAIN as CUSTOM_DOMAIN


# Reverts multi-threshold function operator fusion: Unpacks the operator to the
# naive operator definition: y = sum(weights * (x >= thresholds))
# TODO: Not sure if "inline" is the right term here...
@passes.verify.tolerance
@passes.register("inline-thresholds")
class InlineThresholds(Transformation, RewriteRulePass):
    def pattern(self, op, x, thresholds, weights):
        return op.MultiThreshold(x, thresholds, weights, _domain=CUSTOM_DOMAIN)

    def rewrite(self, op, x, thresholds, weights):
        # Comparison of inputs and all corresponding thresholds: Expand input
        # dimensions to match the threshold parameter shape via broadcasting
        shape = op.Constant(value=ir.tensor([*x.shape, 1], name="shape"))
        steps = op.LessOrEqual(thresholds, op.Reshape(x, shape, allowzero=0))

        # Type-casing turns boolean unit steps to reducible floats followed by
        # weighting for non-unit steps or non-monotonicity
        steps = op.Mul(op.Cast(steps, to=ir.DataType.FLOAT), weights)

        # Finally the multi-threshold output reduces over all steps removing the
        # previously expanded dimension
        axes = op.Constant(value=ir.tensor([-1], name="axes"))
        return op.ReduceSum(steps, axes, keepdims=0)


# Reverts multi-threshold function operator fusion: Unpacks the operator to the
# naive operator definition: y = sum(weights * (x >= thresholds))
# TODO: Not sure if "inline" is the right term here...
@passes.verify.tolerance
@passes.register("inline-thresholds")
class InlineUnitThresholds(Transformation, RewriteRulePass):
    def pattern(self, op, x, thresholds):
        return op.MultiThreshold(x, thresholds, _domain=CUSTOM_DOMAIN)

    def rewrite(self, op, x, thresholds):
        # Comparison of inputs and all corresponding thresholds: Expand input
        # dimensions to match the threshold parameter shape via broadcasting
        shape = op.Constant(value=ir.tensor([*x.shape, 1], name="shape"))
        steps = op.LessOrEqual(thresholds, op.Reshape(x, shape, allowzero=0))

        # Type-casing turns boolean unit steps to reducible floats
        steps = op.Cast(steps, to=ir.DataType.FLOAT)

        # Finally the multi-threshold output reduces over all steps removing the
        # previously expanded dimension
        axes = op.Constant(value=ir.tensor([-1], name="axes"))
        return op.ReduceSum(steps, axes, keepdims=0)
