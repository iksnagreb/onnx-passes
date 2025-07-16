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
class _DistributiveLhs(Transformation, RewriteRulePass):
    __MUL__: callable
    __ADD__: callable

    def rewrite(self, op, x, y, z):
        return self.__MUL__(op, x, self.__ADD__(op, y, z))

    def pattern(self, op, x, y, z):
        return self.__ADD__(op, self.__MUL__(op, x, y), self.__MUL__(op, x, z))


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
class _DistributiveRhs(Transformation, RewriteRulePass):
    __MUL__: callable
    __ADD__: callable

    def rewrite(self, op, x, y, z):
        return self.__MUL__(op, self.__ADD__(op, y, z), x)

    def pattern(self, op, x, y, z):
        return self.__ADD__(op, self.__MUL__(op, y, x), self.__MUL__(op, z, x))


# For commutative mul-like operation there is no distinction between left- and
# right-distributivity, this is simply called *distributivity*
class _Distributive(_DistributiveLhs):
    @property
    def commute(self):
        return True


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


# Involution (self-inverse) template: f(f(x)) = x
class _Involution(Transformation, RewriteRulePass):
    __OP__: callable

    def pattern(self, op, x):
        return self.__OP__(op, self.__OP__(op, x))

    def rewrite(self, op, x):
        return x


# Idempotence template (repeated application has no effect) - there are two
# variants of this, one for unary and one for binary operators:
#   unary: f(f(x)) = f(x), binary: f(x, x) = x
class _Idempotence(Transformation, RewriteRulePass):
    __OP__: callable

    @property
    def arity(self):
        # Note: __OP__ (self, op, ...) -> ??? where arity is the number of ...
        return len(inspect.signature(self.__OP__).parameters) - 1

    def pattern(self, op, x):
        if self.arity == 1:
            return self.__OP__(op, self.__OP__(op, x))
        return self.__OP__(op, x, x)

    def rewrite(self, op, x):
        if self.arity == 1:
            return self.__OP__(op, x)
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


# Annihilator template: f(x, a) = a for some constant a
class _Annihilator(Transformation, RewriteRulePass):
    __OP__: callable
    __ANNIHILATOR__: Any

    def pattern(self, op, x, a):
        return self.__OP__(op, x, a)

    def check(self, op, x, a):
        if x.shape is not None and (a := ir.convenience.get_const_tensor(a)):
            return np.all(a.numpy() == self.__ANNIHILATOR__)
        return False

    def rewrite(self, op, x, a):
        return op.Expand(
            a, op.Constant(value_ints=np.broadcast_shapes(x.shape, a.shape))
        )
