# All algebraic passes are transformations derived from pattern-based rewrite
# rules
from onnx_passes.passes.base import Transformation, RewriteRulePass
# Checking ir.Value for being constants and comparing constants to be identical
from onnx_passes.passes.util import identical_constants, is_constant, is_signed

# Some templates do not fully implement the Transformation or RewriteRulePass
# methods and need to be tagged as ABC
import abc


# Left-distributivity template: x * (y + z) = x * y + x * z
class _DistributiveLhs(Transformation, RewriteRulePass):
    __MUL__: callable
    __ADD__: callable

    def pattern(self, op, x, y, z):
        return self.__MUL__(op, x, self.__ADD__(op, y, z))

    def check(self, op, x, y, z):
        return is_constant(x) and (is_constant(y) or is_constant(z))

    def rewrite(self, op, x, y, z):
        return self.__ADD__(op, self.__MUL__(op, x, y), self.__MUL__(op, x, z))


# Right-distributivity template: (y + z) * x = y * x + z * x
class _DistributiveRhs(Transformation, RewriteRulePass):
    __MUL__: callable
    __ADD__: callable

    def pattern(self, op, x, y, z):
        return self.__MUL__(op, self.__ADD__(op, y, z), x)

    def check(self, op, x, y, z):
        return is_constant(x) and (is_constant(y) or is_constant(z))

    def rewrite(self, op, x, y, z):
        return self.__ADD__(op, self.__MUL__(op, y, x), self.__MUL__(op, z, x))


# For commutative mul-like operation there is no distinction between left- and
# right-distributivity, this is simply called *distributivity*
class _Distributive(_DistributiveLhs):
    @property
    def commute(self):
        return True


# @passes.verify.tolerance
# @passes.register("algebraic")
# class MoveAddPastMul(_Distributive):
#     __MUL__ = lambda _, op, x, y: op.Mul(x, y)
#     __ADD__ = lambda _, op, x, y: op.Add(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class MoveOrPastAnd(_Distributive):
#     __MUL__ = lambda _, op, x, y: op.And(x, y)
#     __ADD__ = lambda _, op, x, y: op.Or(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class MoveAndPastOr(_Distributive):
#     __MUL__ = lambda _, op, x, y: op.Or(x, y)
#     __ADD__ = lambda _, op, x, y: op.And(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class MoveBitwiseOrPastBitwiseAnd(_Distributive):
#     __MUL__ = lambda _, op, x, y: op.BitwiseAnd(x, y)
#     __ADD__ = lambda _, op, x, y: op.BitwiseOr(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class MoveBitwiseAndPastBitwiseOr(_Distributive):
#     __MUL__ = lambda _, op, x, y: op.BitwiseOr(x, y)
#     __ADD__ = lambda _, op, x, y: op.BitwiseAnd(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class MoveMinPastMax(_Distributive):
#     __MUL__ = lambda _, op, x, y: op.Max(x, y)
#     __ADD__ = lambda _, op, x, y: op.Min(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class MoveMaxPastMin(_Distributive):
#     __MUL__ = lambda _, op, x, y: op.Min(x, y)
#     __ADD__ = lambda _, op, x, y: op.Max(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class MoveAddPastMatMulLhs(_DistributiveLhs):
#     __MUL__ = lambda _, op, x, y: op.MatMul(x, y)
#     __ADD__ = lambda _, op, x, y: op.Add(x, y)
#
#
# @passes.verify.tolerance
# @passes.register("algebraic")
# class MoveAddPastMatMulRhs(_DistributiveRhs):
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
