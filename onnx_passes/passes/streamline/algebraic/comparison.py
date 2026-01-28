# ir.Model, ir.DataType, ir.passes.PassResult, ir.from_proto, ir.to_proto, ...
import onnx_ir as ir

# Matching against one value pattern from a selection of alternative patterns
from onnxscript.rewriter.pattern import OrValue

# Algebraic properties as transformation templates
from onnx_passes.passes.streamline.algebraic._properties import _Converse

# All algebraic passes are transformations derived from pattern-based rewrite
# rules
from onnx_passes.passes.base import Transformation, RewriteRuleSetPass
# Checking ir.Value for being constants and comparing constants to be identical
from onnx_passes.passes.util import (
    is_constant, true_like, false_like, zeros_like, ones_like
)

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# Function with partially applied arguments: Used to greate a generator
# mechanism inserting operators into a pattern template
from functools import partial

# Type hinting: Mark template arguments in Transformations templates as Callable
from collections.abc import Callable

# Working with constant tensors/values, and inserting constants such as infinity
import numpy as np


# ==============================================================================
# Transformations derived from templates by specializing basic algebraic
# properties relating comparison operators ==, !=, >, >=, < and <=
# ==============================================================================

# Convert all instances of Less (<) to Greater (>) to reduce the number of rules
# to implement below
@passes.verify.equality
@passes.register("algebraic")
class ConvertLessToGreater(_Converse):
    __OP__ = lambda _, op, x, y: op.Less(x, y)
    __CONVERSE__ = lambda _, op, x, y: op.Greater(x, y)


# Convert all instances of LessOrEqual (<=) to GreaterOrEqual (>=) to reduce the
# number of rules to implement below
@passes.verify.equality
@passes.register("algebraic")
class ConvertLessOrEqualToGreaterOrEqual(_Converse):
    __OP__ = lambda _, op, x, y: op.LessOrEqual(x, y)
    __CONVERSE__ = lambda _, op, x, y: op.GreaterOrEqual(x, y)


# ==============================================================================
# Other properties relating comparison operators ==, !=, >, >=, < and <=: The
# general strategy here is to try reordering and eliminating inputs such that
# the right hand side is all-constant (can be constant-folded into a single
# constant value) while the left hand side remains non-constant
# ==============================================================================

# Eliminates all instances of comparing a tensor to itself, which yields a
# constant, either True (x == x, x <= x, x >= x) or False (x != x, x < x, x > x)
@passes.verify.equality
@passes.register("algebraic")
class EliminateComparison(Transformation, RewriteRuleSetPass):
    def pattern(self):
        return [
            lambda op, x: op.Equal(x, x),
            lambda op, x: op.LessOrEqual(x, x),
            lambda op, x: op.GreaterOrEqual(x, x),
            lambda op, x: op.Less(x, x),
            lambda op, x: op.Greater(x, x),
        ]

    def rewrite(self):
        return [
            true_like, true_like, true_like, false_like, false_like
        ]


# Eliminates instances of comparing a tensor to infinities, which yields a
# constant.
@passes.verify.equality
@passes.register("algebraic")
class EliminateInfinityComparison(Transformation, RewriteRuleSetPass):
    def pattern(self):
        return [
            lambda op, x, a: op.LessOrEqual(x, a),  # x <= +inf == True
            lambda op, x, a: op.GreaterOrEqual(x, a),  # x >= -inf == True
            lambda op, x, a: op.Less(x, a),  # x < -inf == False
            lambda op, x, a: op.Greate(x, a),  # x > +inf == False

            lambda op, x, a: op.Less(a, x),  # +inf < x == False
            lambda op, x, a: op.Greater(a, x),  # -inf > x == False
            lambda op, x, a: op.LessOrEqual(a, x),  # -inf <= x == True
            lambda op, x, a: op.GreaterOrEqual(a, x)  # +inf >= x == True
        ]

    def check(self):
        def _check_lhs(__y__, op, x, a):
            if (a := ir.convenience.get_const_tensor(a)) is not None:
                return np.all(a.numpy() == __y__)
            return False

        def _check_rhs(__y__, op, x, a):
            if (a := ir.convenience.get_const_tensor(a)) is not None:
                return np.all(a.numpy() == __y__)
            return False

        for __y__ in [+np.inf, -np.inf, -np.inf, +np.inf]:
            yield partial(_check_lhs, __y__)

        for __y__ in [+np.inf, -np.inf, -np.inf, +np.inf]:
            yield partial(_check_rhs, __y__)

    def rewrite(self):
        def _rewrite_lhs(__op__, op, x, a):
            return __op__(op, op.Equal(a, x))

        def _rewrite_rhs(__op__, op, x, a):
            return __op__(op, op.Equal(x, a))

        for __OP__ in [true_like, true_like, false_like, false_like]:
            yield partial(_rewrite_lhs, __OP__)

        for __OP__ in [false_like, false_like, true_like, true_like]:
            yield partial(_rewrite_rhs, __OP__)


# A. Same value is added to both sides of a comparison: Remove the addition and
# expand the output to recover the shape after broadcasting...
@passes.verify.tolerance
@passes.register("algebraic")
class EliminateAddFromComparison(Transformation, RewriteRuleSetPass):
    @property
    def commute(self):
        return True

    __OPS__ = [
        lambda op: op.Equal,
        lambda op: op.Less,
        lambda op: op.LessOrEqual,
        lambda op: op.Greater,
        lambda op: op.GreaterOrEqual
    ]

    def pattern(self):
        # Pattern template matches same value added to both sides of the
        # comparison
        def _pattern(__op__, op, x, y, z):
            return __op__(op)(op.Add(x, z), op.Add(y, z))

        # Instantiate the pattern variations for each operator listed above
        for __OP__ in self.__OPS__:
            # Fix the template parameter __OP__
            yield partial(_pattern, __OP__)

    def check(self):
        # Match condition template: Same value added to both sides can always be
        # eliminated
        def _check(__op__, **_):
            return True

        # Instantiate the pattern variations for each operator listed above
        for __OP__ in self.__OPS__:
            # Fix the template parameter __OP__
            yield partial(_check, __OP__)

    def rewrite(self):
        # Rewrite pattern template eliminates the value added to both side and
        # inserts an Expand operation to restore the output shape broadcasting
        def _rewrite(__op__, op, x, y, z):
            return op.Expand(
                __op__(op)(x, y), op.Shape(op.Add(x, op.Add(y, z)))
            )

        # Instantiate the pattern variations for each operator listed above
        for __OP__ in self.__OPS__:
            # Fix the template parameter __OP__
            yield partial(_rewrite, __OP__)


# B. Different values added to either side of a comparison: Rearrange such that
# constants end up on the right side and non-constants on the left...
@passes.verify.tolerance
@passes.register("algebraic")
class GroupAddComparison(Transformation, RewriteRuleSetPass):
    @property
    def commute(self):
        return True

    __OPS__ = [
        lambda op: op.Equal,
        lambda op: op.Less,
        lambda op: op.LessOrEqual,
        lambda op: op.Greater,
        lambda op: op.GreaterOrEqual
    ]

    def pattern(self):
        # Pattern template matches addition on the left hand side of the
        # comparison
        def _pattern_lhs(__op__, op, x, y, z):
            return __op__(op)(op.Add(x, y), z)

        # Pattern template matches addition on the right hand side of the
        # comparison
        def _pattern_rhs(__op__, op, x, y, z):
            return __op__(op)(x, op.Add(y, z))

        # Instantiate the pattern variations for each operator listed above
        for __OP__ in self.__OPS__:
            # Fix the template parameter __OP__
            yield partial(_pattern_lhs, __OP__)
            yield partial(_pattern_rhs, __OP__)

    def check(self):
        # Match condition template: Constants should be on the right,
        # non-constants on the left hand side of the comparison
        def _check_lhs(__op__, op, x, y, z):
            return is_constant(x) or is_constant(y)

        # Match condition template: Constants should be on the right,
        # non-constants on the left hand side of the comparison
        def _check_rhs(__op__, op, x, y, z):
            return not is_constant(y) or not is_constant(z)

        # Instantiate the pattern variations for each operator listed above
        for __OP__ in self.__OPS__:
            # Fix the template parameter __OP__
            yield partial(_check_lhs, __OP__)
            yield partial(_check_rhs, __OP__)

    def rewrite(self):
        # Rewrite template moves constants/non-constants to their respective
        # side
        def _rewrite_lhs(__op__, op, x, y, z):
            # Constness signature of the left hand side selects the replacement
            # pattern
            signature = (is_constant(x), is_constant(y))

            # Zero matching the left hand side type inserted in case both left
            # sides are constants
            zero = op.CastLike(op.Constant(value_float=0.0), x)

            # Select one of three replacement patterns by signature - no need to
            # implement a pattern for (0,0) as this is the identity
            return {
                # x + y == z -> x == z - y
                (0, 1): __op__(op)(x, op.Sub(z, y)),
                # x + y == z -> y == z - x
                (1, 0): __op__(op)(y, op.Sub(z, x)),
                # x + y == z -> 0 == z - (x + y)
                (1, 1): __op__(op)(zero, op.Sub(z, op.Add(x, y))),
            }[signature]

        # Rewrite template moves constants/non-constants to their respective
        # side
        def _rewrite_rhs(__op__, op, x, y, z):
            # Constness signature of the right hand side selects the replacement
            # pattern
            signature = (is_constant(y), is_constant(z))

            # Zero matching the right hand side type inserted in case both right
            # sides are constants
            zero = op.CastLike(op.Constant(value_float=0.0), y)

            # Select one of three replacement patterns by signature - no need to
            # implement a pattern for (1,1) as this is the identity
            return {
                # x == y + z -> x - (y + z) == 0
                (0, 0): __op__(op)(op.Sub(x, op.Add(y, z)), zero),
                # x == y + z -> x - y == z
                (0, 1): __op__(op)(op.Sub(x, y), z),
                # x == y + z -> x - z == y
                (1, 0): __op__(op)(op.Sub(x, z), y),
            }[signature]

        # Instantiate the pattern variations for each operator listed above
        for __OP__ in self.__OPS__:
            # Fix the template parameter __OP__
            yield partial(_rewrite_lhs, __OP__)
            yield partial(_rewrite_rhs, __OP__)


# # C. Absorb addition of a constant value into a comparison with a constant
# # value on either side...
# # TODO: Seems just to be a special case of the more genric reordering above,
# #  just skipping some extra steps when the bare constant is on the left?
# @passes.verify.tolerance
# @passes.register("algebraic")
# class AbsorbAddIntoComparison(Transformation, RewriteRuleSetPass):
#     @property
#     def commute(self):
#         return True
#
#     __OPS__ = [
#         lambda op: op.Equal,
#         lambda op: op.Less,
#         lambda op: op.LessOrEqual,
#         lambda op: op.Greater,
#         lambda op: op.GreaterOrEqual
#     ]
#
#     def pattern(self):
#         # Pattern template matching addition on the left hand side input of
#         # the comparison
#         def _pattern_lhs(__op__, op, x, a, b):
#             return __op__(op)(op.Add(x, a), b)
#
#         # Pattern template matching addition on the right hand side input of
#         # the comparison
#         def _pattern_rhs(__op__, op, x, a, b):
#             return __op__(op)(a, op.Add(x, b))
#
#         # Instantiate the pattern variations for each operator listed above
#         for __OP__ in self.__OPS__:
#             # Fix the template parameter __OP__
#             yield partial(_pattern_lhs, __OP__)
#             yield partial(_pattern_rhs, __OP__)
#
#     def check(self):
#         # Match condition template: Constant on either side, Add commutes - no
#         # need to check, same condition for both sides
#         def _check(__op__, op, x, a, b):
#             return is_constant(a) and is_constant(b)
#
#         # Instantiate the pattern variations for each operator listed above
#         for __OP__ in self.__OPS__:
#             # Fix the template parameter __OP__
#             yield partial(_check, __OP__)
#             yield partial(_check, __OP__)
#
#     def rewrite(self):
#         # Rewrite template absorbs both constants into the right hand side of
#         # the comparison
#         def _rewrite_lhs(__op__, op, x, a, b):
#             return __op__(op)(x, op.Sub(b, a))
#
#         # Rewrite template absorbs both constants into the left hand side of
#         # the comparison
#         def _rewrite_rhs(__op__, op, x, a, b):
#             return __op__(op)(op.Sub(a, b), x)
#
#         # Instantiate the pattern variations for each operator listed above
#         for __OP__ in self.__OPS__:
#             # Fix the template parameter __OP__
#             yield partial(_rewrite_lhs, __OP__)
#             yield partial(_rewrite_rhs, __OP__)


# D. Absorb multiplication of a constant value into a comparison with a constant
# value on either side, if the absorbed constant is positive
# TODO: Handle more generic case of negative constants which flip the comparison
#  to its converse
@passes.verify.tolerance
@passes.register("algebraic")
class AbsorbMulIntoComparison(Transformation, RewriteRuleSetPass):
    @property
    def commute(self):
        return True

    __OPS__ = [
        lambda op: op.Equal,
        lambda op: op.Less,
        lambda op: op.LessOrEqual,
        lambda op: op.Greater,
        lambda op: op.GreaterOrEqual
    ]

    def pattern(self):
        # Pattern template matching multiplication on the left hand side input
        # of the comparison
        def _pattern_lhs(__op__, op, x, a, b):
            return __op__(op)(op.Mul(x, a), b)

        # Pattern template matching multiplication on the right hand side input
        # of the comparison
        def _pattern_rhs(__op__, op, x, a, b):
            return __op__(op)(a, op.Mul(x, b))

        # Instantiate the pattern variations for each operator listed above
        for __OP__ in self.__OPS__:
            # Fix the template parameter __OP__
            yield partial(_pattern_lhs, __OP__)
            yield partial(_pattern_rhs, __OP__)

    def check(self):
        def _check_lhs(__op__, op, x, a, b):
            if is_constant(a) and is_constant(b):
                if np.all(ir.convenience.get_const_tensor(a).numpy() > 0):
                    return True
            return False

        def _check_rhs(__op__, op, x, a, b):
            if is_constant(a) and is_constant(b):
                if np.all(ir.convenience.get_const_tensor(b).numpy() > 0):
                    return True
            return False

        # Instantiate the pattern variations for each operator listed above
        for __OP__ in self.__OPS__:
            # Fix the template parameter __OP__
            yield partial(_check_lhs, __OP__)
            yield partial(_check_rhs, __OP__)

    def rewrite(self):
        # Rewrite template absorbs both constants into the right hand side of
        # the comparison
        def _rewrite_lhs(__op__, op, x, a, b):
            return __op__(op)(x, op.Div(b, a))

        # Rewrite template absorbs both constants into the left hand side of
        # the comparison
        def _rewrite_rhs(__op__, op, x, a, b):
            return __op__(op)(op.Div(a, b), x)

        # Instantiate the pattern variations for each operator listed above
        for __OP__ in self.__OPS__:
            # Fix the template parameter __OP__
            yield partial(_rewrite_lhs, __OP__)
            yield partial(_rewrite_rhs, __OP__)


# Expands a constant of minimum values to the shape and type of the input x
def min_like(op, x):
    # TODO: Actually make use of the minimum of the type...
    return op.CastLike(op.Constant(value_float=-np.inf), x)


# Expands a constant of maximum values to the shape and type of the input x
def max_like(op, x):
    # TODO: Actually make use of the maximum of the type...
    return op.CastLike(op.Constant(value_float=+np.inf), x)


# Absorbs Clip operators into the constant side of a comparison operator by
# applying a generalized inverse of Clip s.t.
#   Clip(x, min, max) == a <=> x == Clip^-1(x, min, max)
@passes.verify.tolerance
@passes.register("algebraic")
class AbsorbClipIntoComparison(Transformation, RewriteRuleSetPass):
    # Generalized inverse of clipping: Pushes out of bounds inputs to the
    # infinities
    @staticmethod
    def __INVERSE__(op, x, _min, _max):
        return op.Where(
            # Within bounds: min <= x <= max
            op.And(op.LessOrEqual(_min, x), op.LessOrEqual(x, _max)),
            # Pass through the within bounds input
            x,
            # Out of bounds: x < min -> -inf, x > max -> +inf
            op.Where(op.Less(x, _min), min_like(op, x), max_like(op, x))
        )

    __OPS__ = [
        lambda op: op.Equal,
        lambda op: op.Less,
        lambda op: op.LessOrEqual,
        lambda op: op.Greater,
        lambda op: op.GreaterOrEqual
    ]

    def pattern(self):
        # Pattern template matches the function applied to the left hand side
        def _pattern_lhs(__op__, op, x, a, _min, _max):
            return __op__(op)(op.Clip(x, _min, _max), a)

        # Pattern template matches the function applied to the left hand side
        def _pattern_rhs(__op__, op, x, a, _min, _max):
            return __op__(op)(a, op.Clip(x, _min, _max))

        # Instantiate the pattern variations for each operator listed above
        for __OP__ in self.__OPS__:
            # Fix the template parameter __OP__
            yield partial(_pattern_lhs, __OP__)
            yield partial(_pattern_rhs, __OP__)

    def check(self):
        # Only rewrite if the graph actually gets simplified, i.e., the right
        # hand side is constant
        def _check_lhs(__op__, op, x, a, _min, _max):
            return (not is_constant(x) and is_constant(a)
                    and is_constant(_min) and is_constant(_max))

        # Only rewrite if the graph actually gets simplified, i.e., the right
        # hand side is constant
        def _check_rhs(__op__, op, x, a, _min, _max):
            return (not is_constant(x) and is_constant(a)
                    and is_constant(_min) and is_constant(_max))

        # Instantiate the pattern variations for each operator listed above
        for __OP__ in self.__OPS__:
            # Fix the template parameter __OP__
            yield partial(_check_lhs, __OP__)
            yield partial(_check_rhs, __OP__)

    def rewrite(self):
        # Rewrite by applying the inverse function to the right hand side and
        # dropping the original function from the left hand side
        def _rewrite_lhs(__op__, op, x, a, _min, _max):
            return __op__(op)(x, self.__INVERSE__(op, a, _min, _max))

        # Rewrite by applying the inverse function to the left hand side and
        # dropping the original function from the right hand side
        def _rewrite_rhs(__op__, op, x, a, _min, _max):
            return __op__(op)(self.__INVERSE__(op, a, _min, _max), x)

        # Instantiate the pattern variations for each operator listed above
        for __OP__ in self.__OPS__:
            # Fix the template parameter __OP__
            yield partial(_rewrite_lhs, __OP__)
            yield partial(_rewrite_rhs, __OP__)


# Some transformation templates rely on inspecting the signature/parameters of
# the operator-specializing function
import inspect

# Custom ONNX operator domain and the unit in the last place (ULP) operator
from onnx_passes.ops import DOMAIN as CUSTOM_DOMAIN
from onnx_passes.ops.ulp import Ulp  # noqa: Used via registry


# Absorbs a function into the constant side of a comparison operator based on a
# generalized inverse of the function, which might insert several branches of
# comparisons, separating increasing from decreasing segments of the function.
#
# TODO: For now, this only supports elementwise unary functions, though some
#  infrastructure for handling n-ary functions is already in place...
class _AbsorbFunctionIntoComparison(Transformation, RewriteRuleSetPass):
    # Placeholder for matching elementwise functions on the non-constant side of
    # the comparison
    __FUNCTION__: Callable

    # Placeholder for the inverse function: Must accept the same arguments as
    # the function and an optional input specifying the branch index
    __INVERSE__: Callable

    # Split the inverse into increasing (+1) and decreasing (-1) branches,
    # where the first entry is the branch index k for evaluating the inverse
    __BRANCHES__: list[tuple[int, bool]] = [(0, +1)]

    @property
    def _arity(self):
        # __FUNCTION__: (self, op, ...) -> ??? where arity is the number of ...
        return len(inspect.signature(self.__FUNCTION__).parameters) - 1

    @property
    def _arity_inverse(self):
        # __INVERSE__: (self, op, ...) -> ??? where arity is the number of ...
        return len(inspect.signature(self.__INVERSE__).parameters) - 1

    @property
    def _accepts_branch_index(self):
        # Check whether the optional branch index is accepted by the inverse
        # placeholder
        return "_branch_index" in inspect.signature(self.__INVERSE__).parameters

    # TODO: To allow for future extension to n-ary functions
    def _function(self, op, x):
        return self.__FUNCTION__(op, x)

    def _inverse(self, op, *args, _branch_index: int = 0):
        if self._accepts_branch_index:
            return self.__INVERSE__(op, *args, _branch_index=_branch_index)
        return self.__INVERSE__(op, *args)

    __OPS__ = [
        lambda op: op.Equal,
        lambda op: op.Less,
        lambda op: op.LessOrEqual,
        lambda op: op.Greater,
        lambda op: op.GreaterOrEqual
    ]

    # Pattern generator: Generates left and right hand side variations of the
    # match pattern for all comparison operators in __OPS__
    def pattern(self):
        def _pattern_lhs(__op__, op, x, a):
            return __op__(op)(self._function(op, x), a)

        def _pattern_rhs(__op__, op, x, a):
            return __op__(op)(a, self._function(op, x))

        for __OP__ in self.__OPS__:
            # Fix the template parameter __OP__
            yield partial(_pattern_lhs, __OP__)
            yield partial(_pattern_rhs, __OP__)

    # Check generator: Generates left and right hand side variations of the
    # match condition for all comparison operators in __OPS__
    def check(self):
        def _check_lhs(__op__, op, x, a):
            return not is_constant(x) and is_constant(a)

        def _check_rhs(__op__, op, x, a):
            return not is_constant(x) and is_constant(a)

        for __OP__ in self.__OPS__:
            # Fix the template parameter __OP__
            yield partial(_check_lhs, __OP__)
            yield partial(_check_rhs, __OP__)

    # Rewrite generator: Generates left and right hand side variations of the
    # replacement pattern for all comparison operators in __OPS__
    def rewrite(self):
        def _rewrite_lhs(__op__, op, x, a):
            # Collect comparison replacement patterns for each branch of the
            # generalized inverse function
            branches = {+1: [], -1: []}

            # Evaluate branches and collect replacement patterns by branch
            # direction
            for k, d in self.__BRANCHES__:
                # Evaluate the kth branch of the inverse on all constant inputs
                branches[d].append(self._inverse(op, a, _branch_index=k))

            # Decreasing branches must be corrected by a small positive amount,
            # the ULP to account for switching > to >= comparisons
            for i, b in enumerate(branches[-1]):
                branches[-1][i] =  op.Add(b, op.Ulp(b, _domain=CUSTOM_DOMAIN))

            # If there are no decreasing branches, joining the branches can be
            # simplified to a single comparison by taking the minimum over the
            # increasing branches
            if not branches[-1]:
                return __op__(op)(x, op.Min(*branches[+1]))

            # If there are no increasing branches, joining the branches can be
            # simplified to a single comparison by taking the maximum over the
            # decreasing branches
            if not branches[+1]:
                return op.Not(__op__(op)(x, op.Max(*branches[-1])))

            # Join the mixed branches of the inverse function disjunctively by a
            # chain of Or functions (unfortunately there is no variadic Or...).
            #
            # Note: The chain of Or with alternating negations can be simplified
            # to the following single-level Xor expression with only a single
            # negation at the output by taking the min/max of the branches.
            return (
                op.Not(
                    op.Xor(
                        __op__(op)(
                            x,
                            op.Max(*branches[-1])
                        ),
                        __op__(op)(
                            x,
                            op.Max(
                                op.Min(*branches[+1]),
                                op.Max(*branches[-1])
                            )
                        )
                    )
                )
            )

        def _rewrite_rhs(__op__, op, x, a):
            # Collect comparison replacement patterns for each branch of the
            # generalized inverse function
            branches = {+1: [], -1: []}

            # Evaluate branches and collect replacement patterns by branch
            # direction
            for k, d in self.__BRANCHES__:
                # Evaluate the kth branch of the inverse on all constant inputs
                branches[d].append(self._inverse(op, a, _branch_index=k))

            # Decreasing branches must be corrected by a small positive amount,
            # the ULP to account for switching > to >= comparisons
            for i, b in enumerate(branches[-1]):
                branches[-1][i] =  op.Add(b, op.Ulp(b, _domain=CUSTOM_DOMAIN))

            # If there are no decreasing branches, joining the branches can be
            # simplified to a single comparison by taking the maximum over the
            # increasing branches
            if not branches[-1]:
                return __op__(op)(op.Max(*branches[+1]), x)

            # If there are no increasing branches, joining the branches can be
            # simplified to a single comparison by taking the minimum over the
            # decreasing branches
            if not branches[+1]:
                return op.Not(__op__(op)(op.Min(*branches[-1])), x)

            # Join the mixed branches of the inverse function disjunctively by a
            # chain of Or functions (unfortunately there is no variadic Or...).
            #
            # Note: The chain of Or with alternating negations can be simplified
            # to the following single-level Xor expression with only a single
            # negation at the output by taking the min/max of the branches.
            return (
                op.Not(
                    op.Xor(
                        __op__(op)(
                            op.Min(*branches[-1]),
                            x
                        ),
                        __op__(op)(
                            op.Min(
                                op.Max(*branches[+1]),
                                op.Min(*branches[-1])
                            ),
                            x
                        )
                    )
                )
            )

        for __OP__ in self.__OPS__:
            # Fix the template parameter __OP__
            yield partial(_rewrite_lhs, __OP__)
            yield partial(_rewrite_rhs, __OP__)


@passes.verify.tolerance
@passes.register("algebraic")
class AbsorbReluIntoComparison(_AbsorbFunctionIntoComparison):
    __FUNCTION__ = lambda _, op, x: op.Relu(x)

    # Generalized inverse of Relu:
    #   Relu^-1(x) = {-inf for x < 0, x for x >= 0}
    @staticmethod
    def __INVERSE__(op, x):
        return op.Where(op.Less(x, zeros_like(op, x)), min_like(op, x), x)


@passes.verify.tolerance
@passes.register("algebraic")
class AbsorbSigmoidIntoComparison(_AbsorbFunctionIntoComparison):
    __FUNCTION__ = lambda _, op, x: op.Sigmoid(x)

    # Generalized inverse of Sigmoid:
    #   Sigmoid^-1(x) = {
    #       -inf for x <= 0, log(x / (1 - x)) for 0 < x < 1, +inf for x >= 1
    #   }
    @staticmethod
    def __INVERSE__(op, x):
        # Sanitize the input x to not evaluate the logarithm on inputs where it
        # is not defined: This should not affect the result but prevents parts
        # of the replacement pattern to be evaluated on illegal inputs
        sanitized_x = op.Where(
            # Check for input inside defined range: 0 < x < 0
            op.And(op.Less(zeros_like(op, x), x), op.Less(x, ones_like(op, x))),
            # Pass through legal inputs
            x,
            # Does not matter, will never actually use the value, just not use
            # >=1.0, <=0 to avoid illegal values or divide by zero...
            op.CastLike(op.Constant(value_float=0.5), x)
        )

        # Select from three cases depending on x: x <= 0, 0 < x < 1, x >= 1
        return op.Where(
            # Check for input inside defined range: 0 < x < 1
            op.And(op.Less(zeros_like(op, x), x), op.Less(x, ones_like(op, x))),
            # Proper inverse of Sigmoid on 0 < x < 1: log(x / (1 - x))
            op.Log(
                op.Div(
                    sanitized_x, op.Sub(ones_like(op, sanitized_x), sanitized_x)
                )
            ),
            # Select positive of negative infinity depending on the side at
            # which the input is out of bounds
            op.Where(
                # Out of range on the lower bound: x <= 0?
                op.LessOrEqual(x, zeros_like(op, x)),
                # Map to negative infinity: -inf for x <= 0
                min_like(op, x),
                # Map to positive infinity: +inf for x >= 1
                max_like(op, x)
            )
        )


# Inverse Silu is defined in the custom domain and needs to be made available as
# an ONNX Script function once used
from onnx_passes.ops.inverse_swish import InverseSilu  # noqa: Used via registry
from onnx_passes.ops.swish import Silu  # noqa: Used via registry


@passes.verify.tolerance
@passes.register("algebraic")
class AbsorbSiluIntoComparison(_AbsorbFunctionIntoComparison):
    # Match the composite representation of the Silu function, as there is no
    # Silu operator in standard ONNX
    __FUNCTION__ = lambda _, op, x: \
        OrValue([x * op.Sigmoid(x), op.Silu(x, _domain=CUSTOM_DOMAIN)])

    # Allow the multiplication of the composite representation to commute so we
    # only have to write this pattern once
    @property
    def commute(self):
        return True

    # Replace by the inverse Silu defined in the custom domain, offering
    # branches to select
    @staticmethod
    def __INVERSE__(op, x, _branch_index: int = 0):
        return op.InverseSilu(x, k=_branch_index, _domain=CUSTOM_DOMAIN)

    # Silu (or rather its inverse) has two branches: A principal where the
    # function is increasing and a secondary where it is decreasing
    __BRANCHES__ = [(0, +1), (-1, -1)]


@passes.verify.tolerance
@passes.register("algebraic")
class AbsorbSquareIntoComparison(_AbsorbFunctionIntoComparison):
    __FUNCTION__ = lambda _, op, x: OrValue([op.Mul(x, x), op.Pow(x, 2)])

    @staticmethod
    def __INVERSE__(op, x, _branch_index: int = 0):
        # Short aliases to infinity and NaN used for out of range inputs
        inf = op.Constant(value_float=float("inf"))

        # Principal branch of the square root: Regular square root to the right
        # of zero, negatives mapped to negative infinity, as x**2 >= -? = True
        if _branch_index == 0:
            return op.Where(
                op.GreaterOrEqual(x, zeros_like(op, x)), op.Sqrt(x), op.Neg(inf)
            )

        # Secondary branch of the square root: Selects the negated square root
        # as x**2 >= a has two solutions {+sqrt(a), -sqrt(a)}
        if _branch_index == -1:
            return op.Where(
                op.GreaterOrEqual(x, zeros_like(op, x)), op.Neg(op.Sqrt(x)), inf
            )

        # Invalid branch selected: Square root has only two branches. Instead of
        # raising an exception, try to stay within ONNX by returning NaN.
        return op.Expand(op.Constant(value_float=np.nan), op.Shape(x))

    # Square (or rather its inverse) has two branches: A principal where the
    # function is increasing and a secondary where it is decreasing
    __BRANCHES__ = [(0, +1), (-1, -1)]


@passes.verify.tolerance
@passes.register("algebraic")
class AbsorbExpIntoComparison(_AbsorbFunctionIntoComparison):
    __FUNCTION__ = lambda _, op, x: op.Exp(x)

    # Generalized inverse of Exp:
    #   Exp^-1(x) = {-inf for x <= 0, Log(x) for x > 0}
    @staticmethod
    def __INVERSE__(op, x):
        return op.Where(
            op.LessOrEqual(x, zeros_like(op, x)), min_like(op, x), op.Log(x)
        )


@passes.verify.tolerance
@passes.register("algebraic")
class AbsorbLogIntoComparison(_AbsorbFunctionIntoComparison):
    __FUNCTION__ = lambda _, op, x: op.Log(x)
    __INVERSE__ = lambda _, op, x: op.Exp(x)
