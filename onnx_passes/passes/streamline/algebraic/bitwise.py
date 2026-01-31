# ir.Value, ir.convenience.get_const_tensor
import onnx_ir as ir

# Algebraic properties as transformation templates
from onnx_passes.passes.streamline.algebraic._properties import (
    _Associative,
    _Commutative,
    _Distributive,
    _Involution,
    _Idempotence,
    _Absorption,
    _Annihilator,
)

# All algebraic passes are transformations derived from pattern-based rewrite
# rules
from onnx_passes.passes.base import Transformation, RewriteRulePass, \
    RewriteRuleSetPass

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# NumPy used during match condition checks to operate on shapes and tensors
import numpy as np


# ==============================================================================
# Transformations derived from templates by specializing basic algebraic
# properties relating bitwise and, or, xor and negation
# ==============================================================================

@passes.verify.equality
@passes.register("algebraic")
class GroupBitwiseOr(_Associative, _Commutative):
    __OP__ = lambda _, op, x, y: op.BitwiseOr(x, y)


@passes.verify.equality
@passes.register("algebraic")
class GroupBitwiseAnd(_Associative, _Commutative):
    __OP__ = lambda _, op, x, y: op.BitwiseAnd(x, y)


@passes.verify.equality
@passes.register("algebraic")
class GroupBitwiseXor(_Associative, _Commutative):
    __OP__ = lambda _, op, x, y: op.BitwiseXor(x, y)


@passes.verify.equality
@passes.register("algebraic")
class DistributiveBitwiseAndBitwiseOr(_Distributive):
    __MUL__ = lambda _, op, x, y: op.BitwiseAnd(x, y)
    __ADD__ = lambda _, op, x, y: op.BitwiseOr(x, y)


@passes.verify.equality
@passes.register("algebraic")
class DistributiveBitwiseOrBitwiseAnd(_Distributive):
    __MUL__ = lambda _, op, x, y: op.BitwiseOr(x, y)
    __ADD__ = lambda _, op, x, y: op.BitwiseAnd(x, y)


@passes.verify.equality
@passes.register("algebraic")
class DistributiveBitwiseAndBitwiseXor(_Distributive):
    __MUL__ = lambda _, op, x, y: op.BitwiseAnd(x, y)
    __ADD__ = lambda _, op, x, y: op.BitwiseXor(x, y)


@passes.verify.equality
@passes.register("algebraic")
class EliminateBitwiseNot(_Involution):
    __OP__ = lambda _, op, x: op.BitwiseNot(x)


@passes.verify.equality
@passes.register("algebraic")
class EliminateBitwiseAnd(_Idempotence):
    __OP__ = lambda _, op, x, y: op.BitwiseAnd(x, y)


@passes.verify.equality
@passes.register("algebraic")
class EliminateBitwiseOr(_Idempotence):
    __OP__ = lambda _, op, x, y: op.BitwiseOr(x, y)


@passes.verify.equality
@passes.register("algebraic")
class EliminateBitwiseXor(_Commutative, Transformation, RewriteRulePass):
    def pattern(self, op, x):
        return op.BitwiseXor(x, x)

    def rewrite(self, op, x):
        return op.Expand(op.CastLike(op.Constant(value_int=0), x))


@passes.verify.equality
@passes.register("algebraic")
class EliminateAbsorptionBitwise(_Absorption, _Commutative):
    __OP1__ = lambda _, op, x, y: op.BitwiseAnd(x, y)
    __OP2__ = lambda _, op, x, y: op.BitwiseOr(x, y)


@passes.verify.equality
@passes.register("algebraic")
class EliminateAnnihilatorBitwiseAnd(_Annihilator, _Commutative):
    __OP__ = lambda _, op, x, y: op.BitwiseAnd(x, y)
    __ANNIHILATOR__ = 0


@passes.verify.equality
@passes.register("algebraic")
class EliminateAnnihilatorBitwiseOr(_Annihilator, _Commutative):
    __OP__ = lambda _, op, x, y: op.BitwiseOr(x, y)
    __ANNIHILATOR__ = ~0  # = 111...1


# ==============================================================================
# Other properties relating bitwise and, or and negation: Complementation and De
# Morgan's laws
# ==============================================================================

# TODO: Extract a _Complementation template from the transformations below which
#  could also be shared by numeric and boolean transformations

@passes.verify.equality
@passes.register("algebraic")
class EliminateComplementationBitwiseAnd(Transformation, RewriteRulePass):
    @property
    def commute(self):
        return True

    def pattern(self, op, x):
        return op.BitwiseAnd(x, op.BitwiseNot(x))

    def rewrite(self, op, x):
        return op.Expand(op.CastLike(op.Constant(value_int=0), x), op.Shape(x))


@passes.verify.equality
@passes.register("algebraic")
class EliminateComplementationBitwiseOr(Transformation, RewriteRulePass):
    @property
    def commute(self):
        return True

    def pattern(self, op, x):
        return op.BitwiseOr(x, op.BitwiseNot(x))

    def rewrite(self, op, x):
        return op.Expand(op.CastLike(op.Constant(value_int=~0), x), op.Shape(x))


@passes.verify.equality
@passes.register("algebraic")
class DeMorganBitwise(Transformation, RewriteRuleSetPass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self):
        return [
            lambda op, x, y: op.BitwiseAnd(op.BitwiseNot(x), op.BitwiseNot(y)),
            lambda op, x, y: op.BitwiseOr(op.BitwiseNot(x), op.BitwiseNot(y)),
        ]

    def rewrite(self):
        return [
            lambda op, x, y: op.BitwiseNot(op.BitwiseOr(x, y)),
            lambda op, x, y: op.BitwiseNot(op.BitwiseAnd(x, y)),
        ]


@passes.verify.equality
@passes.register("algebraic")
class MoveBitwiseNotPastBitwiseXor(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y):
        return op.BitwiseXor(op.BitwiseNot(x), y)

    def rewrite(self, op, x, y):
        return op.BitwiseNot(op.BitwiseXor(x, y))


@passes.verify.equality
@passes.register("algebraic")
class ConvertBitwiseXorToBitwiseNot(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, a):
        return op.BitwiseXor(x, a, _outputs=["_out"])

    def check(self, op, x, a, _out):
        if a := ir.convenience.get_const_tensor(a):
            return _out.shape is not None and np.all(a.numpy() == ~0)
        return False

    def rewrite(self, op, x, a, _out):
        return op.Expand(
            op.BitwiseNot(x), op.Constant(value_ints=list(_out.shape))
        )
