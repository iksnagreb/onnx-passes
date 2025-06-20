# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# Checking ir.Value for being identical constants
from onnx_passes.passes.streamline.util import identical_constants


# Reorders multiplications according to the distributive property to group
# constants next to each other to be picked up by subsequent constant folding
@passes.verify.tolerance
@passes.register("distributive")
@passes.register("distributive-addmul")
class DistributiveAddMul(passes.base.Transformation,
                         passes.base.RewriteRulePass):
    # Addition and multiplication are commutative - pattern can be matched in
    # both directions
    @property
    def commute(self) -> bool:
        return True

    # Match pattern (a * x) + (b * x)
    def pattern(self, op, x, a, b):  # noqa: Signature...
        return op.Add(op.Mul(a, x), op.Mul(b, x))

    # Replacement pattern x * (a + b) regrouping the
    def rewrite(self, op, x, a, b):  # noqa: Signature...
        return op.Mul(x, op.Add(a, b))


# Reorders multiplications according to the distributive property to group
# constants next to each other to be picked up by subsequent constant folding
@passes.verify.tolerance
@passes.register("distributive")
@passes.register("distributive-addmul")
class DistributiveAddMulConst(passes.base.Transformation,
                              passes.base.RewriteRulePass):
    # Addition and multiplication are commutative - pattern can be matched in
    # both directions
    @property
    def commute(self) -> bool:
        return True

    # Match pattern (a * x) + (b * x)
    def pattern(self, op, x, y, a, b):  # noqa: Signature...
        return op.Add(op.Mul(a, x), op.Mul(b, y))

    # Pattern match condition: Checks for identical constants a and b
    def check(self, _, x, y, a, b):  # noqa: Signature...
        return identical_constants(a, b)

    # Replacement pattern x * (a + b) regrouping the
    def rewrite(self, op, x, y, a, b):  # noqa: Signature...
        return op.Mul(a, op.Add(x, y))
