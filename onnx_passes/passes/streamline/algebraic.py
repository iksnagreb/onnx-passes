# ir.Value
import onnx_ir as ir

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# All algebraic passes are transformations derived from pattern-based rewrite
# rules
from onnx_passes.passes.base import Transformation, RewriteRulePass

# Checking ir.Value for being constants and comparing constants to be identical
from onnx_passes.passes.util import identical_constants, is_constant


# Algebraic property: x + y = 2x if x = y, transforms a joining addition into
# multiplication by a constant which can propagate through the graph
# @passes.verify.tolerance
@passes.register("algebraic")
class Algebraic2X(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y):
        return op.Add(x, y)

    def check(self, op, x, y):
        return x == y or identical_constants(x, y)

    def rewrite(self, op, x, y):
        return op.Mul(op.initializer(ir.tensor(2.0), name="_"), x)
