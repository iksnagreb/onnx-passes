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
        return op.Add(x, op.Neg(a))


# Associative property: (x * a) * b = x * (a * b), grouping constants a and b to
# enable constant propagation and fusion
@passes.verify.tolerance
@passes.register("algebraic")
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
        return op.Mul(x, op.Reciprocal(a))


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
