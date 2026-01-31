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
# properties relating boolean and, or, xor and negation
# ==============================================================================

@passes.verify.equality
@passes.register("algebraic")
class GroupOr(_Associative, _Commutative):
    __OP__ = lambda _, op, x, y: op.Or(x, y)


@passes.verify.equality
@passes.register("algebraic")
class GroupAnd(_Associative, _Commutative):
    __OP__ = lambda _, op, x, y: op.And(x, y)


@passes.verify.equality
@passes.register("algebraic")
class GroupXor(_Associative, _Commutative):
    __OP__ = lambda _, op, x, y: op.Xor(x, y)


@passes.verify.equality
@passes.register("algebraic")
class DistributiveAndOr(_Distributive):
    __MUL__ = lambda _, op, x, y: op.And(x, y)
    __ADD__ = lambda _, op, x, y: op.Or(x, y)


@passes.verify.equality
@passes.register("algebraic")
class DistributiveOrAnd(_Distributive):
    __MUL__ = lambda _, op, x, y: op.Or(x, y)
    __ADD__ = lambda _, op, x, y: op.And(x, y)


@passes.verify.equality
@passes.register("algebraic")
class DistributiveAndXor(_Distributive):
    __MUL__ = lambda _, op, x, y: op.And(x, y)
    __ADD__ = lambda _, op, x, y: op.Xor(x, y)


@passes.verify.equality
@passes.register("algebraic")
class EliminateNot(_Involution):
    __OP__ = lambda _, op, x: op.Not(x)


@passes.verify.equality
@passes.register("algebraic")
class EliminateAnd(_Idempotence):
    __OP__ = lambda _, op, x, y: op.And(x, y)


@passes.verify.equality
@passes.register("algebraic")
class EliminateOr(_Idempotence):
    __OP__ = lambda _, op, x, y: op.Or(x, y)


@passes.verify.equality
@passes.register("algebraic")
class EliminateXor(_Commutative, Transformation, RewriteRulePass):
    def pattern(self, op, x):
        return op.Xor(x, x)

    def rewrite(self, op, x):
        return op.Expand(op.CastLike(op.Constant(value_int=0), x), op.Shape(x))


@passes.verify.equality
@passes.register("algebraic")
class EliminateAbsorption(_Absorption, _Commutative):
    __OP1__ = lambda _, op, x, y: op.And(x, y)
    __OP2__ = lambda _, op, x, y: op.Or(x, y)


@passes.verify.equality
@passes.register("algebraic")
class EliminateAnnihilatorAnd(_Annihilator, _Commutative):
    __OP__ = lambda _, op, x, y: op.And(x, y)
    __ANNIHILATOR__ = False


@passes.verify.equality
@passes.register("algebraic")
class EliminateAnnihilatorOr(_Annihilator, _Commutative):
    __OP__ = lambda _, op, x, y: op.Or(x, y)
    __ANNIHILATOR__ = True


# ==============================================================================
# Other properties relating boolean and, or and negation: Complementation and De
# Morgan's laws
# ==============================================================================

# TODO: Extract a _Complementation template from the transformations below which
#  could also be shared by numeric and bitwise transformations

@passes.verify.equality
@passes.register("algebraic")
class EliminateComplementationAnd(Transformation, RewriteRulePass):
    @property
    def commute(self):
        return True

    def pattern(self, op, x):
        return op.And(x, op.Not(x))

    def rewrite(self, op, x):
        return op.Expand(op.CastLike(op.Constant(value_int=0), x), op.Shape(x))


@passes.verify.equality
@passes.register("algebraic")
class EliminateComplementationOr(Transformation, RewriteRulePass):
    @property
    def commute(self):
        return True

    def pattern(self, op, x):
        return op.Or(x, op.Not(x))

    def rewrite(self, op, x):
        return op.Expand(op.CastLike(op.Constant(value_int=1), x), op.Shape(x))


@passes.verify.equality
@passes.register("algebraic")
@passes.register()
class DeMorganBoolean(Transformation, RewriteRuleSetPass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self):
        return [
            lambda op, x, y: op.And(op.Not(x), op.Not(y)),
            lambda op, x, y: op.Or(op.Not(x), op.Not(y)),
        ]

    def rewrite(self):
        return [
            lambda op, x, y: op.Not(op.Or(x, y)),
            lambda op, x, y: op.Not(op.And(x, y)),
        ]


@passes.verify.equality
@passes.register("algebraic")
class MoveNotPastXor(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y):
        return op.Xor(op.Not(x), y)

    def rewrite(self, op, x, y):
        return op.Not(op.Xor(x, y))


@passes.verify.equality
@passes.register("algebraic")
class ConvertXorToNot(Transformation, RewriteRulePass):
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, a):
        return op.Xor(x, a, _outputs=["_out"])

    def check(self, op, x, a, _out):
        if a := ir.convenience.get_const_tensor(a):
            return _out.shape is not None and np.all(a.numpy() == True)
        return False

    def rewrite(self, op, x, a, _out):
        return op.Expand(op.Not(x), op.Constant(value_ints=list(_out.shape)))
