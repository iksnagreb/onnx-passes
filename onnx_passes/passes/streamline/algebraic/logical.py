# ir.Value
import onnx_ir as ir

# Matching against one value pattern from a selection of alternative patterns
from onnxscript.rewriter.pattern import OrValue

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# All algebraic passes are transformations derived from pattern-based rewrite
# rules
from onnx_passes.passes.base import Transformation, RewriteRulePass

# Checking ir.Value for being constants and comparing constants to be identical
from onnx_passes.passes.util import identical_constants, is_constant


# ==============================================================================
# Logical Or, And Not are constrained to (signed) integer input and output
# tensors of the same type. These behave as boolean algebras, i.e., the
# following properties are exploited to simplify expressions, and to group,
# propagate, fuse and eventually eliminate constants:
#
# Associativity, and commutativity, the existence of an identity element,
# annihilators and idempotence.
# ==============================================================================

# Associative property: (x or a) or b = x or (a or b), grouping constants a and
# b to enable constant propagation and fusion
@passes.verify.equality
@passes.register("algebraic")
class GroupConstantOr(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, a, b):
        return op.Or(op.Or(x, a), b)

    def check(self, op, x, a, b):
        return is_constant(a) and is_constant(b)

    def rewrite(self, op, x, a, b):
        return op.Or(x, op.Or(a, b))


# Associative property: (x or a) or y = (x or y) or a, grouping non-constants x
# and y to enable constant propagation and fusion for constant a
@passes.verify.equality
@passes.register("algebraic")
class GroupNonConstantOr(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y, a):
        return op.Or(op.Or(x, a), y)

    def check(self, op, x, y, a):
        return is_constant(a) and not is_constant(x) and not is_constant(y)

    def rewrite(self, op, x, y, a):
        return op.Or(op.Or(x, y), a)


# TODO: Eliminating the annihilator for logical Or, i.e., x | 11...1 = 1 needs
#  broadcasting to match the output shape which should be immediately followed
#  by un-broadcasting optimizations, which are not yet available...


# Idempotence property: x or x = x, two identical inputs (dynamic or constant)
# yield the identity
@passes.verify.equality
@passes.register("algebraic")
class EliminateIdempotenceOr(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y):
        return op.Or(x, y)

    def check(self, op, x, y):
        return x == y or identical_constants(x, y)

    def rewrite(self, op, x, y):
        return x


# Associative property: (x and a) and b = x and (a and b), grouping constants a
# and b to enable constant propagation and fusion
@passes.verify.equality
@passes.register("algebraic")
class GroupConstantAnd(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, a, b):
        return op.And(op.And(x, a), b)

    def check(self, op, x, a, b):
        return is_constant(a) and is_constant(b)

    def rewrite(self, op, x, a, b):
        return op.And(x, op.And(a, b))


# Associative property: (x and a) and y = (x and y) and a, grouping
# non-constants x and y to enable constant propagation and fusion for constant a
@passes.verify.equality
@passes.register("algebraic")
class GroupNonConstantAnd(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y, a):
        return op.And(op.And(x, a), y)

    def check(self, op, x, y, a):
        return is_constant(a) and not is_constant(x) and not is_constant(y)

    def rewrite(self, op, x, y, a):
        return op.And(op.And(x, y), a)


# TODO: Eliminating the annihilator for logical And, i.e., x & 00...0 = 0 needs
#  broadcasting to match the output shape which should be immediately followed
#  by un-broadcasting optimizations, which are not yet available...


# Idempotence property: x and x = x, two identical inputs (dynamic or constant)
# yield the identity
@passes.verify.equality
@passes.register("algebraic")
class EliminateIdempotenceAnd(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y):
        return op.And(x, y)

    def check(self, op, x, y):
        return x == y or identical_constants(x, y)

    def rewrite(self, op, x, y):
        return x


# ==============================================================================
# Logical Or, And and Not are linked via distributivity, absorption and some
# other properties, such as De Morgan's laws...
# ==============================================================================

# Distributive property: ax or by = x(a or b) if x = y, reduces conjunctions
# and, if a and b are constants, allows for further constant propagation/fusion.
@passes.verify.equality
@passes.register("algebraic")
class MoveAndPastOr(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y, a, b):
        # Match constants and implicitly True constant inputs
        return op.Or(
            OrValue([op.And(a, x), x], tag_var="lhs"),
            OrValue([op.And(b, y), y], tag_var="rhs")
        )

    def check(self, op, x, y, a, b, lhs, rhs):
        return x == y or identical_constants(x, y)

    def rewrite(self, op, x, y, a, b, lhs, rhs):
        # Inject explicit Trues for missing constants
        a = [a, op.Constant(value=ir.tensor(True, name="trues"))][lhs]
        b = [b, op.Constant(value=ir.tensor(True, name="trues"))][rhs]
        # Compose pattern: Connect both constant branches, each might simplify
        # to True, make sure to have the correct data type
        return op.And(x, op.Or(op.CastLike(a, x), op.CastLike(b, x)))


# Distributive property: a(x or b) = ax or ab, disjunctions past conjunctions
# enables constant propagation - only makes sense if a and b are constants,
# otherwise the left hand side is preferred to reduce conjunctions.
#
# Note: This, together with various associativity rules nicely groups constant
# And and Or nodes to be fused.
@passes.verify.equality
@passes.register("algebraic")
class MoveOrPastAnd(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, a, b):
        return op.And(a, op.Or(x, b))

    def check(self, op, x, a, b):
        return is_constant(a) and is_constant(b)

    def rewrite(self, op, x, a, b):
        return op.Or(op.And(a, x), op.And(a, b))


# Absorption property: x and (x or y) = x, reduces two-input joining pattern to
# identity in the first, independent of the second input
@passes.verify.equality
@passes.register("algebraic")
class EliminateAbsorptionAndOr(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y):
        return op.And(x, op.Or(x, y))

    def rewrite(self, op, x, y):
        return x


# Absorption property: x or (x and y) = x, reduces two-input joining pattern to
# identity in the first, independent of the second input
@passes.verify.equality
@passes.register("algebraic")
class EliminateAbsorptionOrAnd(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y):
        return op.Or(x, op.And(x, y))

    def rewrite(self, op, x, y):
        return x


# De Morgan's law: (not x) and (not y) = not (x or y), propagates Not downstream
# through the graph
@passes.verify.equality
@passes.register("algebraic")
class DeMorganAnd(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y):
        return op.And(op.Not(x), op.Not(y))

    def rewrite(self, op, x, y):
        return op.Not(op.Or(x, y))


# De Morgan's law: (not x) or (not y) = not (x and y), propagates Not downstream
# through the graph
@passes.verify.equality
@passes.register("algebraic")
class DeMorganOr(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y):
        return op.Or(op.Not(x), op.Not(y))

    def rewrite(self, op, x, y):
        return op.Not(op.And(x, y))


# Double negation law: not not x = x, eliminates/fuses aggregated negations
@passes.verify.equality
@passes.register("algebraic")
class EliminateDoubleNot(Transformation, RewriteRulePass):
    def pattern(self, op, x):
        return op.Not(op.Not(x))

    def rewrite(self, op, x):
        return x

# TODO: Eliminating the complementation for logical Or and And, i.e.,
#  x | ~x = 11...1 and x & ~x = 00...0 needs broadcasting to match the output
#  shape which should be immediately followed by un-broadcasting optimizations,
#  which are not yet available...
