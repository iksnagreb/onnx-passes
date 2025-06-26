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
from onnx_passes.passes.util import identical_constants, is_constant, is_signed

# NumPy used during match condition checks to operate on shapes and tensors
import numpy as np


# ==============================================================================
# Addition and Multiplication are constrained to numeric input and output
# tensors of the same type. Despite floating-point arithmetic not strictly being
# associative, it is assumed that these approximately behave as commutative
# groups, i.e., the following properties are exploited to simplify expressions,
# and to group, propagate, fuse and eventually eliminate constants:
#
# Associativity, the existence of an inverse element (additive only for signed,
# multiplicative only for floating-point), the existence of an identity element,
# and commutativity.
#
# As floating-point arithmetic is only approximately associative, all these
# transformations must be tagged @passes.verify.tolerance instead of equality.
# ==============================================================================

# Associative property: (x + a) + b = x + (a + b), grouping constants a and b to
# enable constant propagation and fusion
@passes.verify.tolerance
@passes.register("algebraic")
@passes.register("associative")
@passes.register("associative-add")
class GroupConstantAdd(Transformation, RewriteRulePass):
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
@passes.register("associative-add")
class GroupNonConstantAdd(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y, a):
        return op.Add(op.Add(x, a), y)

    def check(self, op, x, y, a):
        return is_constant(a) and not is_constant(x) and not is_constant(y)

    def rewrite(self, op, x, y, a):
        return op.Add(op.Add(x, y), a)


# Inverse property: x - x = 0, two identical inputs (dynamic or constant) are
# reduced to a constant zero operator
@passes.verify.tolerance
@passes.register("algebraic")
class EliminateIdentitySub(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y):
        return op.Sub(x, y)

    def check(self, op, x, y):
        return x == y or identical_constants(x, y)

    def rewrite(self, op, x, y):
        return op.CastLike(op.Constant(value=ir.tensor(0, name="zero")), x)


# Inverse property: x - a = x + (-a), simplify subtraction to addition of the
# inverse for signed numeric tensors
@passes.verify.tolerance
@passes.register("algebraic")
class ConvertSubToAdd(Transformation, RewriteRulePass):
    def pattern(self, op, x, a):
        return op.Sub(x, a)

    def check(self, op, x, a):
        return is_constant(a) and is_signed(a.dtype)

    def rewrite(self, op, x, a):
        # Create a constant operator producing the inverse of a with the type
        # matching the other input x: Type-cast to avoid issues due to
        # typ-promotion, such as implicit float->double...
        return op.Add(x, op.CastLike(
            op.Constant(value=ir.tensor(- a.const_value.numpy())), x
        ))


# Associative property: (x * a) * b = x * (a * b), grouping constants a and b to
# enable constant propagation and fusion
@passes.verify.tolerance
@passes.register("algebraic")
@passes.register("associative")
@passes.register("associative-mul")
class GroupConstantMul(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, a, b):
        return op.Mul(op.Mul(x, a), b)

    def check(self, op, x, a, b):
        return is_constant(a) and is_constant(b)

    def rewrite(self, op, x, a, b):
        return op.Mul(x, op.Mul(a, b))


# Associative property: x * (y * a) = (x * y) * a, grouping non-constants x and
# y to enable constant propagation and fusion for constant a
@passes.verify.tolerance
@passes.register("algebraic")
@passes.register("associative")
@passes.register("associative-mul")
class GroupNonConstantMul(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y, a):
        return op.Mul(x, op.Mul(y, a))

    def check(self, op, x, y, a):
        return is_constant(a) and not is_constant(x) and not is_constant(y)

    def rewrite(self, op, x, y, a):
        return op.Mul(op.Mul(x, y), a)


# Inverse property: x / x = 1, two identical inputs (dynamic or constant) are
# reduced to a constant one operator
@passes.verify.tolerance
@passes.register("algebraic")
class EliminateIdentityDiv(Transformation, RewriteRulePass):
    def pattern(self, op, x, y):
        return op.Div(x, y)

    def check(self, op, x, y):
        return x == y or identical_constants(x, y)

    def rewrite(self, op, x, y):
        return op.CastLike(op.Constant(value=ir.tensor(1, name="one")), x)


# Inverse property: x / a = x * (1/a), simplify division to multiplication of
# the inverse for floating-point numeric tensors
@passes.verify.tolerance
@passes.register("algebraic")
class ConvertDivToMul(Transformation, RewriteRulePass):
    def pattern(self, op, x, a):
        return op.Div(x, a)

    # There is no multiplicative inverse of zero, i.e., reject 1/0
    def check(self, op, x, a):
        if v := ir.convenience.get_const_tensor(a):
            return a.dtype.is_floating_point() and np.all(v.numpy() != 0)
        return False

    def rewrite(self, op, x, a):
        # Create a constant operator producing the inverse of a with the type
        # matching the other input x: Type-cast to avoid issues due to
        # typ-promotion, such as implicit float->double...
        return op.Mul(x, op.CastLike(
            op.Constant(value=ir.tensor(1 / a.const_value.numpy())), x
        ))


# ==============================================================================
# Addition and Multiplication are linked via distributivity and some other
# properties, such as expressing repeated addition as multiplication...
# ==============================================================================

# Distributive property: ax + by = x(a + b) if x = y, reduces multiplications
# and, if a and b are constants, allows for further constant propagation/fusion.
#
# Note: With x = y and a = b = 1 this naturally yields expressing repeated
# addition as constant multiplication
@passes.verify.tolerance
@passes.register("algebraic")
@passes.register("distributive")
@passes.register("distributive-mul-past-add")
class MoveMulPastAdd(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y, a, b):
        # Compose pattern: Match all variations of a and b which might be
        # implicit or explicit identities:
        return op.Add(
            OrValue([a * x, 1 * x, x], tag_var="lhs"),
            OrValue([b * y, 1 * y, y], tag_var="rhs")
        )

    def check(self, op, x, y, a, b, lhs, rhs):
        return x == y or identical_constants(x, y)

    def rewrite(self, op, x, y, a, b, lhs, rhs):
        # Inject explicit ones for both, originally explicit and implicit
        # identities
        a = [a, op.Constant(value_float=1.0), op.Constant(value_float=1.0)][lhs]
        b = [b, op.Constant(value_float=1.0), op.Constant(value_float=1.0)][rhs]
        # Compose pattern: Connect both constant branches, each might simplify
        # to one, make sure to have the correct data type
        return op.Mul(x, op.Add(op.CastLike(a, x), op.CastLike(b, x)))


# Distributive property: a(x + b) = ax + ab, additions past multiplications
# enables constant propagation - only makes sense if a and b are constants,
# otherwise the left hand side is preferred to reduce multiplications.
#
# Note: This, together with various associativity rules nicely groups constant
# Mul and Add nodes to be fused.
@passes.verify.tolerance
@passes.register("algebraic")
@passes.register("distributive")
@passes.register("distributive-add-past-mul")
class MoveAddPastMul(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, a, b):
        return op.Mul(a, op.Add(x, b))

    def check(self, op, x, a, b):
        return is_constant(a) and is_constant(b)

    def rewrite(self, op, x, a, b):
        return op.Add(op.Mul(a, x), op.Mul(a, b))


# ==============================================================================
# Other properties relating addition, subtraction, multiplication, division and
# some unary operators...
# ==============================================================================

# Anti-commutativity of subtraction: -(x - y) = y - x, simplify expression by
# swapping and eliminating a negation operator
@passes.verify.tolerance
@passes.register("algebraic")
class SwapAntiCommutativeSub(Transformation, RewriteRulePass):
    def pattern(self, op, x, y):
        return op.Neg(op.Sub(x, y))

    # TODO: Should this only be allowed for non-constant y?

    def rewrite(self, op, x, y):
        return op.Sub(y, x)


# Double negation law: --x = x, eliminates/fuses aggregated negations
@passes.verify.tolerance
@passes.register("algebraic")
class EliminateDoubleNeg(Transformation, RewriteRulePass):
    def pattern(self, op, x):
        return op.Neg(op.Neg(x))

    def rewrite(self, op, x):
        return x


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
@passes.verify.tolerance
@passes.register("algebraic")
@passes.register("associative")
@passes.register("associative-logical-or")
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
@passes.verify.tolerance
@passes.register("algebraic")
@passes.register("associative")
@passes.register("associative-logical-or")
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
@passes.verify.tolerance
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
@passes.verify.tolerance
@passes.register("algebraic")
@passes.register("associative")
@passes.register("associative-logical-and")
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
@passes.verify.tolerance
@passes.register("algebraic")
@passes.register("associative")
@passes.register("associative-logical-and")
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
@passes.verify.tolerance
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
@passes.verify.tolerance
@passes.register("algebraic")
@passes.register("distributive")
@passes.register("distributive-and-past-or")
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
@passes.verify.tolerance
@passes.register("algebraic")
@passes.register("distributive")
@passes.register("distributive-or-past-and")
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
@passes.verify.tolerance
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
@passes.verify.tolerance
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
@passes.verify.tolerance
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
@passes.verify.tolerance
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
@passes.verify.tolerance
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
