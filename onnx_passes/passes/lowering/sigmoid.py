# Transformation passes derived from pattern-based rewrite rules
from onnx_passes.passes.base import Transformation, RewriteRulePass

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes


@passes.verify.tolerance
@passes.register("lower-sigmoid")
class LowerSigmoid(Transformation, RewriteRulePass):
    def pattern(self, op, x):
        return op.Sigmoid(x)

    def rewrite(self, op, x):
        # Constant one matching the input to improve readability
        _1 = op.CastLike(op.Constant(value_float=1.0), x)
        # Replacement pattern: 1 / (1 + exp(-x))
        return op.Div(_1, op.Add(_1, op.Exp(op.Neg(x))))
