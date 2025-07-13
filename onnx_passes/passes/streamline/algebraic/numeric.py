# ir.Value
import onnx_ir as ir

# Algebraic properties as transformation templates
from onnx_passes.passes.streamline.algebraic._properties import (
    _Associative,
    _Commutative,
    _Distributive,
    _Involution,
    _Idempotence,
    _Absorption,
    _Annihilator
)

# All algebraic passes are transformations derived from pattern-based rewrite
# rules
from onnx_passes.passes.base import Transformation, RewriteRulePass

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# NumPy used during match condition checks to operate on shapes and tensors
import numpy as np


# ==============================================================================
# Transformations derived from templates by specializing basic algebraic
# properties relating multiplication, addition and negation (+ some unary ops)
# ==============================================================================

@passes.verify.tolerance
@passes.register("algebraic")
class GroupAdd(_Associative, _Commutative):
    __OP__ = lambda _, op, x, y: op.Add(x, y)


@passes.verify.tolerance
@passes.register("algebraic")
class GroupMul(_Associative, _Commutative):
    __OP__ = lambda _, op, x, y: op.Mul(x, y)


@passes.verify.tolerance
@passes.register("algebraic")
class GroupMax(_Associative, _Commutative):
    __OP__ = lambda _, op, x, y: op.Max(x, y)


@passes.verify.tolerance
@passes.register("algebraic")
class GroupMin(_Associative, _Commutative):
    __OP__ = lambda _, op, x, y: op.Min(x, y)


# TODO: Bring back special cases of implicitly 1 constant multiplications...
@passes.verify.tolerance
@passes.register("algebraic")
class DistributiveMulAdd(_Distributive):
    __MUL__ = lambda _, op, x, y: op.Mul(x, y)
    __ADD__ = lambda _, op, x, y: op.Add(x, y)


@passes.verify.tolerance
@passes.register("algebraic")
class DistributiveMaxMin(_Distributive):
    __MUL__ = lambda _, op, x, y: op.Max(x, y)
    __ADD__ = lambda _, op, x, y: op.Min(x, y)


@passes.verify.tolerance
@passes.register("algebraic")
class DistributiveMinMax(_Distributive):
    __MUL__ = lambda _, op, x, y: op.Min(x, y)
    __ADD__ = lambda _, op, x, y: op.Max(x, y)


@passes.verify.tolerance
@passes.register("algebraic")
class EliminateNeg(_Involution):
    __OP__ = lambda _, op, x: op.Neg(x)


@passes.verify.tolerance
@passes.register("algebraic")
class EliminateReciprocal(_Involution):
    __OP__ = lambda _, op, x: op.Reciprocal(x)


@passes.verify.tolerance
@passes.register("algebraic")
class EliminateAbs(_Idempotence):
    __OP__ = lambda _, op, x: op.Abs(x)


@passes.verify.tolerance
@passes.register("algebraic")
class EliminateCeil(_Idempotence):
    __OP__ = lambda _, op, x: op.Ceil(x)


@passes.verify.tolerance
@passes.register("algebraic")
class EliminateFloor(_Idempotence):
    __OP__ = lambda _, op, x: op.Floor(x)


@passes.verify.tolerance
@passes.register("algebraic")
class EliminateRound(_Idempotence):
    __OP__ = lambda _, op, x: op.Round(x)


@passes.verify.tolerance
@passes.register("algebraic")
class EliminateAbsorptionMinMax(_Absorption, _Commutative):
    __OP1__ = lambda _, op, x, y: op.Min(x, y)
    __OP2__ = lambda _, op, x, y: op.Max(x, y)


@passes.verify.tolerance
@passes.register("algebraic")
class EliminateAnnihilatorMul(_Annihilator, _Commutative):
    __OP__ = lambda _, op, x, y: op.Mul(x, y)
    __ANNIHILATOR__ = 0


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

    def rewrite(self, op, x, y):
        return op.Sub(y, x)


@passes.verify.tolerance
@passes.register("algebraic")
class ConvertSubToAdd(Transformation, RewriteRulePass):
    def pattern(self, op, x, y):
        return op.Sub(x, y)

    def check(self, op, x, y):
        return y.dtype is not None and y.dtype.is_signed()

    def rewrite(self, op, x, y):
        return op.Add(x, op.Neg(y))


@passes.verify.tolerance
@passes.register("algebraic")
class ConvertDivToMul(Transformation, RewriteRulePass):
    def pattern(self, op, x, y):
        return op.Div(x, y)

    def check(self, op, x, y):
        return y.dtype is not None and y.dtype.is_floating_point()

    def rewrite(self, op, x, y):
        return op.Mul(x, op.Reciprocal(y))


@passes.verify.tolerance
@passes.register("algebraic")
class ConvertNegToMul(Transformation, RewriteRulePass):
    def pattern(self, op, x):
        return op.Neg(x)

    def rewrite(self, op, x):
        return op.Mul(op.CastLike(op.Constant(value_int=-1), x), x)


# TODO: Extract a _Complementation template from the transformations below which
#  could also be shared by bitwise and boolean transformations

@passes.verify.tolerance
@passes.register("algebraic")
class EliminateComplementationDiv(Transformation, RewriteRulePass):
    def pattern(self, op, x):
        return op.Div(x, x)

    def rewrite(self, op, x):
        return op.Expand(op.CastLike(op.Constant(value_int=1), x), op.Shape(x))


@passes.verify.tolerance
@passes.register("algebraic")
class EliminateComplementationMul(Transformation, RewriteRulePass):
    @property
    def commute(self):
        return True

    def pattern(self, op, x):
        return op.Mul(x, op.Reciprocal(x))

    def rewrite(self, op, x):
        return op.Expand(op.CastLike(op.Constant(value_int=1), x), op.Shape(x))


@passes.verify.tolerance
@passes.register("algebraic")
class EliminateComplementationSub(Transformation, RewriteRulePass):
    def pattern(self, op, x):
        return op.Sub(x, x)

    def rewrite(self, op, x):
        return op.Expand(op.CastLike(op.Constant(value_int=0), x), op.Shape(x))


@passes.verify.tolerance
@passes.register("algebraic")
class EliminateComplementationAdd(Transformation, RewriteRulePass):
    @property
    def commute(self):
        return True

    def pattern(self, op, x, a):
        return op.Add(x, op.Mul(a, x))

    def check(self, op, x, a):
        if v := ir.convenience.get_const_tensor(a):
            return np.all(v.numpy() == -1)
        return False

    def rewrite(self, op, x):
        return op.Expand(op.CastLike(op.Constant(value_int=0), x), op.Shape(x))
