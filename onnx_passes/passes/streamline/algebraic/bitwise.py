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
# BitwiseOr, BitwiseAnd and BitwiseNot are constrained to (signed) integer input
# and output tensors of the same type. These behave as boolean algebras, i.e.,
# the following properties are exploited to simplify expressions, and to group,
# propagate, fuse and eventually eliminate constants:
#
# Associativity, and commutativity, the existence of an identity element,
# annihilators and idempotence.
# ==============================================================================

# Associative property: (x | a) | b = x | (a | b), grouping constants a and b to
# enable constant propagation and fusion
@passes.verify.tolerance
@passes.register("algebraic")
@passes.register("associative")
@passes.register("associative-bitwise-or")
class GroupConstantBitwiseOr(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, a, b):
        return op.BitwiseOr(op.BitwiseOr(x, a), b)

    def check(self, op, x, a, b):
        return is_constant(a) and is_constant(b)

    def rewrite(self, op, x, a, b):
        return op.BitwiseOr(x, op.BitwiseOr(a, b))


# Associative property: (x | a) | y = (x | y) | a, grouping non-constants x and
# y to enable constant propagation and fusion for constant a
@passes.verify.tolerance
@passes.register("algebraic")
@passes.register("associative")
@passes.register("associative-bitwise-or")
class GroupNonConstantBitwiseOr(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y, a):
        return op.BitwiseOr(op.BitwiseOr(x, a), y)

    def check(self, op, x, y, a):
        return is_constant(a) and not is_constant(x) and not is_constant(y)

    def rewrite(self, op, x, y, a):
        return op.BitwiseOr(op.BitwiseOr(x, y), a)


# TODO: Eliminating the annihilator for BitwiseOr, i.e., x | 11...1 = 1 needs
#  broadcasting to match the output shape which should be immediately followed
#  by un-broadcasting optimizations, which are not yet available...


# Idempotence property: x | x = x, two identical inputs (dynamic or constant)
# yield the identity
@passes.verify.tolerance
@passes.register("algebraic")
class EliminateIdempotenceBitwiseOr(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y):
        return op.BitwiseOr(x, y)

    def check(self, op, x, y):
        return x == y or identical_constants(x, y)

    def rewrite(self, op, x, y):
        return x


# Associative property: (x & a) & b = x & (a & b), grouping constants a and b to
# enable constant propagation and fusion
@passes.verify.tolerance
@passes.register("algebraic")
@passes.register("associative")
@passes.register("associative-bitwise-and")
class GroupConstantBitwiseAnd(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, a, b):
        return op.BitwiseAnd(op.BitwiseAnd(x, a), b)

    def check(self, op, x, a, b):
        return is_constant(a) and is_constant(b)

    def rewrite(self, op, x, a, b):
        return op.BitwiseAnd(x, op.BitwiseAnd(a, b))


# Associative property: (x & a) & y = (x & y) & a, grouping non-constants x and
# y to enable constant propagation and fusion for constant a
@passes.verify.tolerance
@passes.register("algebraic")
@passes.register("associative")
@passes.register("associative-bitwise-and")
class GroupNonConstantBitwiseAnd(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y, a):
        return op.BitwiseAnd(op.BitwiseAnd(x, a), y)

    def check(self, op, x, y, a):
        return is_constant(a) and not is_constant(x) and not is_constant(y)

    def rewrite(self, op, x, y, a):
        return op.BitwiseAnd(op.BitwiseAnd(x, y), a)


# TODO: Eliminating the annihilator for BitwiseAnd, i.e., x & 00...0 = 0 needs
#  broadcasting to match the output shape which should be immediately followed
#  by un-broadcasting optimizations, which are not yet available...


# Idempotence property: x & x = x, two identical inputs (dynamic or constant)
# yield the identity
@passes.verify.tolerance
@passes.register("algebraic")
class EliminateIdempotenceBitwiseAnd(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y):
        return op.BitwiseAnd(x, y)

    def check(self, op, x, y):
        return x == y or identical_constants(x, y)

    def rewrite(self, op, x, y):
        return x


# ==============================================================================
# BitwiseOr, BitwiseAnd and BitwiseNot are linked via distributivity, absorption
# and some other properties, such as De Morgan's laws...
# ==============================================================================

# Distributive property: ax | by = x(a | b) if x = y, reduces conjunctions
# and, if a and b are constants, allows for further constant propagation/fusion.
@passes.verify.tolerance
@passes.register("algebraic")
@passes.register("distributive")
@passes.register("distributive-and-past-or")
class MoveBitwiseAndPastBitwiseOr(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y, a, b):
        # Match constants and implicitly 1s constant inputs
        return op.Or(
            OrValue([op.BitwiseOr(a, x), x], tag_var="lhs"),
            OrValue([op.BitwiseOr(b, y), y], tag_var="rhs")
        )

    def check(self, op, x, y, a, b, lhs, rhs):
        return x == y or identical_constants(x, y)

    def rewrite(self, op, x, y, a, b, lhs, rhs):
        # Inject explicit 1s for missing constants
        a = [a, op.Constant(value=ir.tensor(~0, name="ones"))][lhs]
        b = [b, op.Constant(value=ir.tensor(~0, name="ones"))][rhs]
        # Compose pattern: Connect both constant branches, each might simplify
        # to 1s, make sure to have the correct data type
        return op.BitwiseAnd(
            x, op.BitwiseOr(op.CastLike(a, x), op.CastLike(b, x))
        )


# Distributive property: a(x | b) = ax | ab, disjunctions past conjunctions
# enables constant propagation - only makes sense if a and b are constants,
# otherwise the left hand side is preferred to reduce conjunctions.
#
# Note: This, together with various associativity rules nicely groups constant
# And and Or nodes to be fused.
@passes.verify.tolerance
@passes.register("algebraic")
@passes.register("distributive")
@passes.register("distributive-or-past-and")
class MoveBitwiseOrPastBitwiseAnd(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, a, b):
        return op.BitwiseAnd(a, op.BitwiseOr(x, b))

    def check(self, op, x, a, b):
        return is_constant(a) and is_constant(b)

    def rewrite(self, op, x, a, b):
        return op.BitwiseOr(op.BitwiseAnd(a, x), op.BitwiseAnd(a, b))


# Absorption property: x & (x | y) = x, reduces two-input joining pattern to
# identity in the first, independent of the second input
@passes.verify.tolerance
@passes.register("algebraic")
class EliminateAbsorptionBitwiseAndBitwiseOr(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y):
        return op.BitwiseAnd(x, op.BitwiseOr(x, y))

    def rewrite(self, op, x, y):
        return x


# Absorption property: x | (x & y) = x, reduces two-input joining pattern to
# identity in the first, independent of the second input
@passes.verify.tolerance
@passes.register("algebraic")
class EliminateAbsorptionBitwiseOrBitwiseAnd(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y):
        return op.BitwiseOr(x, op.BitwiseAnd(x, y))

    def rewrite(self, op, x, y):
        return x


# De Morgan's law: ~x & ~y = ~(x | y), propagates BitwiseNot downstream through
# the graph
@passes.verify.tolerance
@passes.register("algebraic")
class DeMorganBitwiseAnd(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y):
        return op.BitwiseAnd(op.BitwiseNot(x), op.BitwiseNot(y))

    def rewrite(self, op, x, y):
        return op.BitwiseNot(op.BitwiseOr(x, y))


# De Morgan's law: ~x | ~y = ~(x & y), propagates BitwiseNot downstream through
# the graph
@passes.verify.tolerance
@passes.register("algebraic")
class DeMorganBitwiseOr(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y):
        return op.BitwiseOr(op.BitwiseNot(x), op.BitwiseNot(y))

    def rewrite(self, op, x, y):
        return op.BitwiseNot(op.BitwiseAnd(x, y))


# Double negation law: ~~x = x, eliminates/fuses aggregated bitwise negations
@passes.verify.tolerance
@passes.register("algebraic")
class EliminateDoubleBitwiseNot(Transformation, RewriteRulePass):
    def pattern(self, op, x):
        return op.BitwiseNot(op.BitwiseNot(x))

    def rewrite(self, op, x):
        return x

# TODO: Eliminating the complementation for BitwiseOr and BitwiseAnd, i.e.,
#  x | ~x = 11...1 and x & ~x = 00...0 needs broadcasting to match the output
#  shape which should be immediately followed by un-broadcasting optimizations,
#  which are not yet available...
