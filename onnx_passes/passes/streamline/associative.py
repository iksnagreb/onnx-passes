# ir.Value
import onnx_ir as ir

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# Checking ir.Value for being constants
from onnx_passes.passes.streamline.util import is_constant


# Reorders multiplications according to the associative property to group
# constants next to each other to be picked up by subsequent constant folding
@passes.verify.tolerance
@passes.register("associative")
@passes.register("associative-mul")
class AssociativeMul(passes.base.Transformation, passes.base.RewriteRulePass):
    # Multiplication is commutative - pattern can be matched in both directions
    @property
    def commute(self) -> bool:
        return True

    # Match pattern ((x * a) * b) where a and b are constants
    def pattern(self, op, x, a, b):  # noqa: Signature...
        return op.Mul(op.Mul(x, a), b)

    # Pattern match conditions checking constant parameters - either
    # initializers or const_value must be present
    def check(self, _, x, a, b):  # noqa: Signature...
        return is_constant(a) and is_constant(b)

    # Replacement pattern  (x * (a * b)) regrouping the two constants a and b
    # next to each other
    def rewrite(self, op, x, a, b):  # noqa: Signature...
        return op.Mul(x, op.Mul(a, b))


# Reorders additions according to the associative property to group constants
# next to each other to be picked up by subsequent constant folding
@passes.verify.tolerance
@passes.register("associative")
@passes.register("associative-add")
class AssociativeAdd(passes.base.Transformation, passes.base.RewriteRulePass):
    # Addition is commutative - pattern can be matched in both directions
    @property
    def commute(self) -> bool:
        return True

    # Match pattern ((x + a) + b) where a and b are constants
    def pattern(self, op, x, a, b):  # noqa: Signature...
        return op.Add(op.Add(x, a), b)

    # Pattern match conditions checking constant parameters - either
    # initializers or const_value must be present
    def check(self, _, x, a, b):  # noqa: Signature...
        return is_constant(b) and is_constant(b)

    # Replacement pattern  (x + (a + b)) regrouping the two constants a and b
    # next to each other
    def rewrite(self, op, x, a, b):  # noqa: Signature...
        return op.Add(x, op.Add(a, b))


# Reorders additions according to the associative property to group constants
# next to each other to be picked up by subsequent constant folding
@passes.verify.tolerance
@passes.register("associative")
@passes.register("associative-add")
class AssociativeAddMany(passes.base.Transformation,
                         passes.base.RewriteRulePass):
    # Addition is commutative - pattern can be matched in both directions
    @property
    def commute(self) -> bool:
        return True

    # Match pattern ((a + x) + (b + y)) where a and b are constants
    def pattern(self, op, x, y, a, b):  # noqa: Signature...
        return op.Add(op.Add(a, x), op.Add(b, y))

    # Pattern match conditions checking constant parameters - either
    # initializers or const_value must be present
    def check(self, _, x, y, a, b):  # noqa: Signature...
        return is_constant(a) and is_constant(b)

    # Replacement pattern  ((x + y) + (a + b)) regrouping the two constants a
    # and b next to each other
    def rewrite(self, op, x, y, a, b):  # noqa: Signature...
        return op.Add(op.Add(x, y), op.Add(a, b))


# Reorders additions according to the associative property to group constants
# next to each other to be picked up by subsequent constant folding
@passes.verify.tolerance
@passes.register("associative")
@passes.register("associative-add")
class AssociativeAddSelf(passes.base.Transformation,
                         passes.base.RewriteRulePass):
    # Addition is commutative - pattern can be matched in both directions
    @property
    def commute(self) -> bool:
        return True

    # Match pattern x + (x + a) where x is added to itself
    def pattern(self, op, x, a):  # noqa: Signature...
        return op.Add(x, op.Add(x, a))

    # Replacement pattern 2 * x + a replacing the addition of x + x by a
    # constant multiplication
    def rewrite(self, op, x, a):  # noqa: Signature...
        return op.Add(op.Mul(x, op.initializer(ir.tensor(2.0), name="b")), a)
