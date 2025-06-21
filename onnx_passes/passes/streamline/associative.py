# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# All algebraic associative passes are transformations derived from
# pattern-based rewrite rules
from onnx_passes.passes.base import Transformation, RewriteRulePass

# Checking ir.Value for being constants and comparing constants to be identical
from onnx_passes.passes.streamline.util import is_constant


# Associative property: (x + a) + b = x + (a + b), grouping constants a and b to
# enable constant propagation and fusion
@passes.verify.tolerance
@passes.register("algebraic")
@passes.register("associative")
class AssociativeAdd(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, a, b):
        return op.Add(op.Add(x, a), b)

    def check(self, op, x, a, b):
        return is_constant(a) and is_constant(b)

    def rewrite(self, op, x, a, b):
        return op.Add(x, op.Add(a, b))


# Associative property: (x + a) + y = (x + y) + a, grouping non-constants x and
# y to enable constant propagation and fusion for constant a
@passes.verify.tolerance
@passes.register("algebraic")
@passes.register("associative")
class AssociativeAddJoin(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y, a):
        return op.Add(op.Add(x, a), y)

    def check(self, op, x, y, a):
        return is_constant(a) and not is_constant(x) and not is_constant(y)

    def rewrite(self, op, x, y, a):
        return op.Add(op.Add(x, y), a)


# Associative property: (x * a) * b = x * (a * b), grouping constants a and b to
# enable constant propagation and fusion
@passes.verify.tolerance
@passes.register("algebraic")
@passes.register("associative")
class AssociativeMul(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, a, b):
        return op.Mul(op.Mul(x, a), b)

    def check(self, op, x, a, b):
        return is_constant(a) and is_constant(b)

    def rewrite(self, op, x, a, b):
        return op.Mul(x, op.Mul(a, b))


# Associative property: (x * a) * y = (x * y) * a, grouping non-constants x and
# y to enable constant propagation and fusion for constant a
@passes.verify.tolerance
@passes.register("algebraic")
@passes.register("associative")
class AssociativeMulJoin(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y, a):
        return op.Mul(op.Mul(x, a), y)

    def check(self, op, x, y, a):
        return is_constant(a) and not is_constant(x) and not is_constant(y)

    def rewrite(self, op, x, y, a):
        return op.Mul(op.Mul(x, y), a)
