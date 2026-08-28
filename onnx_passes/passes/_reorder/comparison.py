from onnx_passes.passes._base import (
    RewriteRule, RewriteRuleSetTemplate, Transformation, Sequential
)
from onnx_passes.passes._verify import Verify, tolerance

from onnxscript.rewriter.pattern import OrValue

import onnx_ir as ir
import numpy as np


class ReorderTransitiveEqual_v1(RewriteRule, Verify):
    """Reorder conjunction of equality comparison to move constants right."""

    @staticmethod
    def pattern(op, x, a, b):
        return op.And(op.Equal(x, a), op.Equal(x, b))

    @staticmethod
    def check(context, x, a, b):
        if ir.convenience.get_const_tensor(a) is not None:
            return ir.convenience.get_const_tensor(b) is not None
        return False

    @staticmethod
    def rewrite(op, x, a, b):
        return op.And(op.Equal(x, a), op.Equal(a, b))

    @property
    def commute(self) -> bool:
        return True


@tolerance
class SortConstantComparison_v1(RewriteRuleSetTemplate, Verify):
    """Sort likewise constant comparison over commutative (boolean) connectives.

    For likewise comparison <=> with a common input x and constants a, b, c,
    after sorting: a <= b <= c: (x <=> a) . (x <=> b) . (x <=> c)

    Note: This also assumes associativity to apply sorting to nested connected
    comparisons, i.e., to sort (x >= a) . ((a >= b) . y).
    """

    patterns = (
        lambda op: (op.GreaterOrEqual, op.Xor),
        lambda op: (op.Greater, op.Xor),
        lambda op: (op.Equal, op.Xor),
        lambda op: (op.Less, op.Xor),
        lambda op: (op.LessOrEqual, op.Xor),
        # Note: Greater/Less(OrEqual) connected via Or/And simplify as follows:
        #   x >= a | x >= b -> a >= min(a,b)
        lambda op: (op.Equal, op.Or),
        lambda op: (op.Equal, op.And),
    )

    @staticmethod
    def pattern(partial, op, x, y, a, b):
        comparison, connective = partial(op)

        # Match two options of the pattern, the base case (1), and the
        # potentially recursive case (2), continuing to sort inward.
        #   (1) (x <=> a) .  (x <=> b)
        #   (2) (x <=> a) . ((x <=> b) . y...)
        return connective(
            comparison(x, a),
            OrValue([
                comparison(x, b),
                # Recursive application along associative chains or termination
                # in some other connective or constant.
                connective(comparison(x, b), y),
                # Allow the recursive part (or termination) to commute, pushing
                # the y chain inward with the replacement pattern.
                connective(y, comparison(x, b)),
            ])
        )

    @staticmethod
    def check(op, x, y, a, b):
        if (a := ir.convenience.get_const_tensor(a)) is not None:
            if (b := ir.convenience.get_const_tensor(b)) is not None:
                # Rewrite if any pair of elements is out of order
                return np.any(a.numpy() > b.numpy())
        return False

    @staticmethod
    def rewrite(partial, op, x, y, a, b):
        comparison, connective = partial(op)

        # This is one step of bubblesort swapping neighboring elements. One pass
        # is done reaching the base case where y is None. Sorting is done if the
        # check yields False for all a,b pairs along the chain.
        a = ir.convenience.get_const_tensor(a).numpy()  # noqa: never None
        b = ir.convenience.get_const_tensor(b).numpy()  # noqa: never None

        a, b = np.minimum(a, b), np.maximum(a, b)

        rhs = comparison(x, op.Constant(value=ir.tensor(b)))

        if y is not None:
            rhs = connective(rhs, y)

        return connective(
            comparison(
                x,
                op.Constant(value=ir.tensor(a))
            ),
            rhs
        )


@tolerance
class SortConstantComparison_v2(RewriteRuleSetTemplate, Verify):
    """Sort likewise constant comparison over commutative (boolean) connectives.

    For likewise comparison <=> with a common input x and constants c1 and c2
    and constant boolean coefficients a1 and a2 the constants and coefficients
    are swapped, such that after sorting:

    a1 >= a2 and a1 == a2 -> c1 <= c2: a1(x >= c1) . a2(x >= c2)

    Note: This also assumes associativity to apply sorting to nested connected
    comparisons, i.e., to sort a1(x >= c1) . (a2(x >= c2) . (a3(x >= c3) . y)).
    """

    patterns = (
        lambda op: (op.GreaterOrEqual, op.Xor),
        lambda op: (op.Greater, op.Xor),
        lambda op: (op.Equal, op.Xor),
        lambda op: (op.Less, op.Xor),
        lambda op: (op.LessOrEqual, op.Xor),

        lambda op: (op.GreaterOrEqual, op.Or),
        lambda op: (op.Greater, op.Or),
        lambda op: (op.Equal, op.Or),
        lambda op: (op.Less, op.Or),
        lambda op: (op.LessOrEqual, op.Or),

        lambda op: (op.GreaterOrEqual, op.And),
        lambda op: (op.Greater, op.And),
        lambda op: (op.Equal, op.And),
        lambda op: (op.Less, op.And),
        lambda op: (op.LessOrEqual, op.And),
    )

    @staticmethod
    def pattern(partial, op, x, y, a1, a2, c1, c2):
        comparison, connective = partial(op)

        # Match two options of the pattern, the base case (1), and the
        # potentially recursive case (2), continuing to sort inward.
        #   (1) a1(x <=> c1) .  a2(x <=> c2)
        #   (2) a1(x <=> c1) . (a2(x <=> c2) . y...)
        return connective(
            OrValue([
                comparison(x, c1),
                # Allow coefficients to commute but not the connective, as
                # sorting is with respect to the connective
                op.And(a1, comparison(x, c1)),
                op.And(comparison(x, c1), a1)
            ]),
            OrValue([
                rhs := OrValue([
                    comparison(x, c2),
                    # Allow coefficients to commute but not the connective, as
                    # sorting is with respect to the connective
                    op.And(a2, comparison(x, c2)),
                    op.And(comparison(x, c2), a2)
                ]),
                # Recursive application along associative chains or termination
                # in some other connective or constant.
                connective(rhs, y),
                # Allow the recursive part (or termination) to commute, pushing
                # the y chain inward with the replacement pattern.
                connective(y, rhs)
            ])
        )

    @staticmethod
    def check(context, x, y, a1, a2, c1, c2):
        # Coefficients a1, a2 are optional, if not present they are implicitly
        # assumed to be True as And(True, x) = x.
        if a1 is not None:
            if (a1 := ir.convenience.get_const_tensor(a1)) is not None:
                a1 = a1.numpy()
            else:
                return False
        else:
            a1 = np.asarray(True)

        if a2 is not None:
            if (a2 := ir.convenience.get_const_tensor(a2)) is not None:
                a2 = a2.numpy()
            else:
                return False
        else:
            a2 = np.asarray(True)

        # Comparison constants are not optional, both must be present to decide
        # whether we need to continue swapping neighboring comparisons.
        if (c1 := ir.convenience.get_const_tensor(c1)) is None:
            return False

        if (c2 := ir.convenience.get_const_tensor(c2)) is None:
            return False

        # Swap neighbors if any coefficients are out of order, or if comparisons
        # for matching coefficients are out of order.
        return np.any((a1 < a2) | (a1 == a2) & (c1.numpy() > c2.numpy()))

    @staticmethod
    def rewrite(partial, op, x, y, a1, a2, c1, c2):
        comparison, connective = partial(op)

        # Coefficients a1, a2 are optional, if not present they are implicitly
        # assumed to be True as And(True, x) = x. Comparison constants are not
        # optional and always present as guaranteed by the match condition.
        if a1 is not None:
            if (a1 := ir.convenience.get_const_tensor(a1)) is not None:
                a1 = a1.numpy()
            else:
                a1 = np.asarray(True)
        else:
            a1 = np.asarray(True)

        if a2 is not None:
            if (a2 := ir.convenience.get_const_tensor(a2)) is not None:
                a2 = a2.numpy()
            else:
                a2 = np.asarray(True)
        else:
            a2 = np.asarray(True)

        c1 = ir.convenience.get_const_tensor(c1).numpy()  # noqa: not None
        c2 = ir.convenience.get_const_tensor(c2).numpy()  # noqa: not None

        # This is one step of bubblesort swapping neighboring elements. One pass
        # is done reaching the base case where y is None. Sorting is done if the
        # check yields False for all (a1,a2),(c1,c2) pairs along the chain.
        rhs = op.And(
            comparison(
                x,
                op.Constant(
                    value=ir.tensor(
                        np.where(
                            a1 == a2, np.maximum(c1, c2), np.where(
                                a1 < a2, c1, c2
                            )
                        )
                    )
                )
            ),
            op.Constant(value=ir.tensor(np.minimum(a1, a2)))
        )

        if y is not None:
            rhs = connective(rhs, y)

        return connective(
            op.And(
                comparison(
                    x,
                    op.Constant(
                        value=ir.tensor(
                            np.where(
                                a1 == a2, np.minimum(c1, c2), np.where(
                                    a1 < a2, c2, c1
                                )
                            )
                        )
                    )
                ),
                op.Constant(value=ir.tensor(np.maximum(a1, a2)))
            ),
            rhs
        )


class ReorderConstantComparison_v1(RewriteRuleSetTemplate, Verify):
    """Reorder comparison to a constant to have the constant on the right."""

    patterns = (
        lambda op: (op.GreaterOrEqual, op.Greater),
        lambda op: (op.Greater, op.GreaterOrEqual),
        lambda op: (op.LessOrEqual, op.Less),
        lambda op: (op.Less, op.LessOrEqual),
        lambda op: (op.Equal, op.Equal)
    )

    @staticmethod
    def pattern(pattern, op, x, y):
        return pattern(op)[0](x, y)

    @staticmethod
    def check(context, x, y):
        if ir.convenience.get_const_tensor(x) is not None:
            return ir.convenience.get_const_tensor(y) is None
        return False

    @staticmethod
    def rewrite(pattern, op, x, y):
        return op.Not(pattern(op)[1](y, x))


class ReorderTernaryComparison_v1(RewriteRuleSetTemplate, Verify):
    """Reorder ternary conditional following likewise comparisons."""

    patterns = (
        lambda op: op.Equal,
        lambda op: op.Greater,
        lambda op: op.GreaterOrEqual,
        lambda op: op.Less,
        lambda op: op.LessOrEqual,
    )

    @staticmethod
    def pattern(partial, op, x, condition, a, b):
        return op.Where(condition, partial(op)(x, a), partial(op)(x, b))

    @staticmethod
    def rewrite(partial, op, x, condition, a, b):
        return partial(op)(x, op.Where(condition, a, b))


class ReorderComparisonLoop_v1(Sequential, Transformation):
    """Exhaustively apply comparison reordering transformations."""

    passes = [
        ReorderTransitiveEqual_v1,
        SortConstantComparison_v1,
        SortConstantComparison_v2,
        ReorderConstantComparison_v1,
        ReorderTernaryComparison_v1,
    ]

    exhaustive = True
