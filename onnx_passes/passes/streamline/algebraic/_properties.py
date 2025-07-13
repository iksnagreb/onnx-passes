# ir.Model, ir.passes.PassResult, ir.from_proto, ir.to_proto, ...
import onnx_ir as ir

# All algebraic passes are transformations derived from pattern-based rewrite
# rules
from onnx_passes.passes.base import (
    Transformation, RewriteRulePass, RewriteRuleSetPass
)
# Checking ir.Value for being constants and comparing constants to be identical
from onnx_passes.passes.util import identical_constants, is_constant, is_signed

# Type annotation matching anything, used for annihilator constant placeholder
from typing import Any

# Some templates do not fully implement the Transformation or RewriteRulePass
# methods and need to be tagged as ABC
import abc

# Some transformation templates rely on inspecting the signature/parameters of
# the operator-specializing function
import inspect

# NumPy used during match condition checks to operate on shapes and tensors
import numpy as np


# # Left-distributivity template: x * (y + z) = x * y + x * z
# class _DistributiveLhs(Transformation, RewriteRulePass):
#     __MUL__: callable
#     __ADD__: callable
#
#     def pattern(self, op, x, y, z):
#         return self.__MUL__(op, x, self.__ADD__(op, y, z))
#
#     def check(self, op, x, y, z):
#         return is_constant(x) and (is_constant(y) or is_constant(z))
#
#     def rewrite(self, op, x, y, z):
#         return self.__ADD__(op, self.__MUL__(op, x, y), self.__MUL__(op, x,z))


# Left-distributivity template: x * (y + z) = x * y + x * z
class _DistributiveLhs(Transformation, RewriteRuleSetPass):
    __MUL__: callable
    __ADD__: callable

    def _lhs(self, op, x, y, z):
        return self.__MUL__(op, x, self.__ADD__(op, y, z))

    def _rhs(self, op, x, y, z):
        return self.__ADD__(op, self.__MUL__(op, x, y), self.__MUL__(op, x, z))

    def pattern(self):
        return [self._lhs, self._rhs]

    def check(self):
        return [
            lambda _, x, y, z: \
                is_constant(x) and (is_constant(y) or is_constant(z)),
            lambda _, x, y, z: \
                is_constant(x) and not (is_constant(y) or is_constant(z))
        ]

    def rewrite(self):
        return [self._rhs, self._lhs]


# # Right-distributivity template: (y + z) * x = y * x + z * x
# class _DistributiveRhs(Transformation, RewriteRulePass):
#     __MUL__: callable
#     __ADD__: callable
#
#     def pattern(self, op, x, y, z):
#         return self.__MUL__(op, self.__ADD__(op, y, z), x)
#
#     def check(self, op, x, y, z):
#         return is_constant(x) and (is_constant(y) or is_constant(z))
#
#     def rewrite(self, op, x, y, z):
#         return self.__ADD__(op, self.__MUL__(op, y, x), self.__MUL__(op, z,x))


# Right-distributivity template: (y + z) * x = y * x + z * x
class _DistributiveRhs(Transformation, RewriteRuleSetPass):
    __MUL__: callable
    __ADD__: callable

    def _lhs(self, op, x, y, z):
        return self.__MUL__(op, self.__ADD__(op, y, z), x)

    def _rhs(self, op, x, y, z):
        return self.__ADD__(op, self.__MUL__(op, y, x), self.__MUL__(op, z, x))

    def pattern(self):
        return [self._lhs, self._rhs]

    def check(self):
        return [
            lambda _, x, y, z: \
                is_constant(x) and (is_constant(y) or is_constant(z)),
            lambda _, x, y, z: \
                is_constant(x) and not (is_constant(y) or is_constant(z))
        ]

    def rewrite(self):
        return [self._rhs, self._lhs]


# For commutative mul-like operation there is no distinction between left- and
# right-distributivity, this is simply called *distributivity*
class _Distributive(_DistributiveLhs):
    @property
    def commute(self):
        return True


# @passes.verify.tolerance
# @passes.register("algebraic")
# class DistributiveMulAdd(_Distributive):
#     __MUL__ = lambda _, op, x, y: op.Mul(x, y)
#     __ADD__ = lambda _, op, x, y: op.Add(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class DistributiveAndOr(_Distributive):
#     __MUL__ = lambda _, op, x, y: op.And(x, y)
#     __ADD__ = lambda _, op, x, y: op.Or(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class DistributiveOrAnd(_Distributive):
#     __MUL__ = lambda _, op, x, y: op.Or(x, y)
#     __ADD__ = lambda _, op, x, y: op.And(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class DistributiveAndXor(_Distributive):
#     __MUL__ = lambda _, op, x, y: op.And(x, y)
#     __ADD__ = lambda _, op, x, y: op.Xor(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class DistributiveBitwiseAndBitwiseOr(_Distributive):
#     __MUL__ = lambda _, op, x, y: op.BitwiseAnd(x, y)
#     __ADD__ = lambda _, op, x, y: op.BitwiseOr(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class DistributiveBitwiseOrBitwiseAnd(_Distributive):
#     __MUL__ = lambda _, op, x, y: op.BitwiseOr(x, y)
#     __ADD__ = lambda _, op, x, y: op.BitwiseAnd(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class DistributiveBitwiseAndBitwiseXor(_Distributive):
#     __MUL__ = lambda _, op, x, y: op.BitwiseAnd(x, y)
#     __ADD__ = lambda _, op, x, y: op.BitwiseXor(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class DistributiveMaxMin(_Distributive):
#     __MUL__ = lambda _, op, x, y: op.Max(x, y)
#     __ADD__ = lambda _, op, x, y: op.Min(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class DistributiveMinMax(_Distributive):
#     __MUL__ = lambda _, op, x, y: op.Min(x, y)
#     __ADD__ = lambda _, op, x, y: op.Max(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class DistributiveMatMulAddLhs(_DistributiveLhs):
#     __MUL__ = lambda _, op, x, y: op.MatMul(x, y)
#     __ADD__ = lambda _, op, x, y: op.Add(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class DistributiveMatMulAddRhs(_DistributiveRhs):
#     __MUL__ = lambda _, op, x, y: op.MatMul(x, y)
#     __ADD__ = lambda _, op, x, y: op.Add(x, y)


# Commutativity template: x + y = y + x
class _Commutative(Transformation, RewriteRulePass, abc.ABC):
    @property
    def commute(self):
        return True


# Associativity template: (x + y) + z = x + (y + z)
class _Associative(Transformation, RewriteRulePass):
    __OP__: callable

    def pattern(self, op, x, y, z):
        return self.__OP__(op, self.__OP__(op, x, y), z)

    def check(self, op, x, y, z):
        # 1. Group two constants if there is one non-constant input
        if not is_constant(x) and is_constant(y) and is_constant(z):
            return True
        # 2. Group two non-constants if there is one constant input
        if is_constant(x) and not is_constant(y) and not is_constant(z):
            return True
        # 3. Do not change the grouping of all constant or all non-constant
        return False

    def rewrite(self, op, x, y, z):
        return self.__OP__(op, x, self.__OP__(op, y, z))


# @passes.verify.tolerance
# @passes.register("algebraic")
# class GroupAdd(_Associative, _Commutative):
#     __OP__ = lambda _, op, x, y: op.Add(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class GroupMul(_Associative, _Commutative):
#     __OP__ = lambda _, op, x, y: op.Mul(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class GroupBitwiseOr(_Associative, _Commutative):
#     __OP__ = lambda _, op, x, y: op.BitwiseOr(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class GroupBitwiseAnd(_Associative, _Commutative):
#     __OP__ = lambda _, op, x, y: op.BitwiseAnd(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class GroupBitwiseXor(_Associative, _Commutative):
#     __OP__ = lambda _, op, x, y: op.BitwiseXor(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class GroupOr(_Associative, _Commutative):
#     __OP__ = lambda _, op, x, y: op.Or(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class GroupAnd(_Associative, _Commutative):
#     __OP__ = lambda _, op, x, y: op.And(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class GroupXor(_Associative, _Commutative):
#     __OP__ = lambda _, op, x, y: op.Xor(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class GroupMax(_Associative, _Commutative):
#     __OP__ = lambda _, op, x, y: op.Max(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class GroupMin(_Associative, _Commutative):
#     __OP__ = lambda _, op, x, y: op.Min(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class GroupMatMul(_Associative):
#     __OP__ = lambda _, op, x, y: op.MatMul(x, y)


# Involution (self-inverse) template: f(f(x)) = x
class _Involution(Transformation, RewriteRulePass):
    __OP__: callable

    def pattern(self, op, x):
        return self.__OP__(self.__OP__(x))

    def rewrite(self, op, x):
        return x


# @passes.verify.tolerance
# @passes.register("algebraic")
# class EliminateNot(_Involution):
#     __OP__ = lambda _, op, x: op.Not(x)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class EliminateBitwiseNot(_Involution):
#     __OP__ = lambda _, op, x: op.BitwiseNot(x)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class EliminateNeg(_Involution):
#     __OP__ = lambda _, op, x: op.Neg(x)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class EliminateReciprocal(_Involution):
#     __OP__ = lambda _, op, x: op.Reciprocal(x)


# Idempotence template (repeated application has no effect) - there are two
# variants of this, one for unary and one for binary operators:
#   unary: f(f(x)) = f(x), binary: f(x, x) = x
class _Idempotence(Transformation, RewriteRulePass):
    __OP__: callable

    @property
    def arity(self):
        # Note: __OP__ (self, op, ...) -> ??? where arity is the number of ...
        return len(inspect.signature(self.__OP__).parameters) - 2

    def pattern(self, op, x):
        if self.arity == 1:
            return self.__OP__(self.__OP__(x))
        return self.__OP__(x, x)

    def rewrite(self, op, x):
        if self.arity == 1:
            return self.__OP__(x)
        return x


# # Idempotence binary operator template: f(x, x) = x
# class _IdempotenceBinary(Transformation, RewriteRulePass):
#     __OP__: callable
#
#     def pattern(self, op, x):
#         return self.__OP__(x, x)
#
#     def rewrite(self, op, x):
#         return x


# @passes.verify.tolerance
# @passes.register("algebraic")
# class EliminateAbs(_Idempotence):
#     __OP__ = lambda _, op, x: op.Abs(x)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class EliminateCeil(_Idempotence):
#     __OP__ = lambda _, op, x: op.Ceil(x)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class EliminateFloor(_Idempotence):
#     __OP__ = lambda _, op, x: op.Floor(x)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class EliminateRound(_Idempotence):
#     __OP__ = lambda _, op, x: op.Round(x)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class EliminateAnd(_Idempotence):
#     __OP__ = lambda _, op, x, y: op.And(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class EliminateOr(_Idempotence):
#     __OP__ = lambda _, op, x, y: op.Or(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class EliminateBitwiseAnd(_Idempotence):
#     __OP__ = lambda _, op, x, y: op.BitwiseAnd(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class EliminateBitwiseOr(_Idempotence):
#     __OP__ = lambda _, op, x, y: op.BitwiseOr(x, y)


# Absorption law template: x OP1 (x OP2 y) = x OP2 (x OP1 y) = x
class _Absorption(Transformation, RewriteRuleSetPass):
    __OP1__: callable
    __OP2__: callable

    def pattern(self):
        return [
            lambda op, x, y: self.__OP1__(op, x, self.__OP2__(op, x, y)),
            lambda op, x, y: self.__OP2__(op, x, self.__OP1__(op, x, y)),
        ]

    def rewrite(self):
        return [
            lambda op, x, y: x,
            lambda op, x, y: x,
        ]


# @passes.verify.tolerance
# @passes.register("algebraic")
# class EliminateAbsorptionBoolean(_Absorption, _Commutative):
#     __OP1__ = lambda _, op, x, y: op.And(x, y)
#     __OP2__ = lambda _, op, x, y: op.Or(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class EliminateAbsorptionBitwise(_Absorption, _Commutative):
#     __OP1__ = lambda _, op, x, y: op.BitwiseAnd(x, y)
#     __OP2__ = lambda _, op, x, y: op.BitwiseOr(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class EliminateAbsorptionMinMax(_Absorption, _Commutative):
#     __OP1__ = lambda _, op, x, y: op.Min(x, y)
#     __OP2__ = lambda _, op, x, y: op.Max(x, y)


# Annihilator template: f(x, a) = a for some constant a
class _Annihilator(Transformation, RewriteRulePass):
    __OP__: callable
    __ANNIHILATOR__: Any

    def pattern(self, op, x, a):
        return self.__OP__(op, x, a)

    def check(self, op, x, a):
        if a := ir.convenience.get_const_tensor(a):
            return np.all(a.numpy() == self.__ANNIHILATOR__)
        return False

    def rewrite(self, op, x, a):
        return op.Expand(a, op.Shape(self.__OP__(op, x, a)))

# @passes.verify.tolerance
# @passes.register("algebraic")
# class EliminateAnnihilatorAnd(_Annihilator, _Commutative):
#     __OP__ = lambda _, op, x, y: op.And(x, y)
#     __ANNIHILATOR__ = False
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class EliminateAnnihilatorOr(_Annihilator, _Commutative):
#     __OP__ = lambda _, op, x, y: op.Or(x, y)
#     __ANNIHILATOR__ = True
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class EliminateAnnihilatorBitwiseAnd(_Annihilator, _Commutative):
#     __OP__ = lambda _, op, x, y: op.BitwiseAnd(x, y)
#     __ANNIHILATOR__ = 0
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class EliminateAnnihilatorBitwiseOr(_Annihilator, _Commutative):
#     __OP__ = lambda _, op, x, y: op.BitwiseOr(x, y)
#     __ANNIHILATOR__ = ~0  # = 111...1
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class EliminateAnnihilatorMul(_Annihilator, _Commutative):
#     __OP__ = lambda _, op, x, y: op.Mul(x, y)
#     __ANNIHILATOR__ = 0
