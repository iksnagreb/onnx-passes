# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# All algebraic distributive passes are transformations derived from
# pattern-based rewrite rules
from onnx_passes.passes.base import Transformation, RewriteRulePass

# Checking ir.Value for being constants and comparing constants to be identical
from onnx_passes.passes.streamline.util import identical_constants, is_constant


# Distributive property: ax + bx = x(a + b), reduces multiplications and, if a
# and b are constants, enables further constant propagation/fusion.
@passes.verify.tolerance
@passes.register("algebraic")
@passes.register("distributive")
class DistributiveAXAddBX(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, a, b):
        return op.Add(op.Mul(a, x), op.Mul(b, x))

    def rewrite(self, op, x, a, b):
        return op.Mul(x, op.Add(a, b))


# Distributive property: ax + by = x(a + b) if x = y, reduces multiplications
# and, if a and b are constants, allows for further constant propagation/fusion.
@passes.verify.tolerance
@passes.register("algebraic")
@passes.register("distributive")
class DistributiveAXAddBY(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y, a, b):
        return op.Add(op.Mul(a, x), op.Mul(b, y))

    def check(self, op, x, y, a, b):
        return x == y or identical_constants(x, y)

    def rewrite(self, op, x, y, a, b):
        return op.Mul(x, op.Add(a, b))


# Distributive property: a(x + b) = ax + ab, additions past multiplications
# enables constant propagation - only makes sense if a and b are constants,
# otherwise the left hand side is preferred to reduce multiplications.
@passes.verify.tolerance
@passes.register("algebraic")
@passes.register("distributive")
class DistributiveAddPastMul(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, a, b):
        return op.Mul(a, op.Add(x, b))

    def check(self, op, x, a, b):
        return is_constant(a) and is_constant(b)

    def rewrite(self, op, x, a, b):
        return op.Add(op.Mul(a, x), op.Mul(a, b))
