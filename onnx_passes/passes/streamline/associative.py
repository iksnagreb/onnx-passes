# ir.Model, ir.passes.PassResult, ir.from_proto, ir.to_proto, ...
import onnx_ir as ir

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes


# Checks whether the ir.Value represents a constant: Either is_initializer or
# has a const_value set
def is_constant(v: ir.Value):
    return v.const_value is not None or v.is_initializer()


# Reorders multiplications according to the associative law to group constants
# next to each other to be picked up by subsequent constant folding
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


# Reorders additions according to the associative law to group constants
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
        return is_constant(a) and is_constant(b)

    # Replacement pattern  (x + (a + b)) regrouping the two constants a and b
    # next to each other
    def rewrite(self, op, x, a, b):  # noqa: Signature...
        return op.Add(x, op.Add(a, b))
