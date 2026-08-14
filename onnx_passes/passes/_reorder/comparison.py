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

        # Match (x <=> a) . (x <=> b), as well as (x <=> a) . ((x <=> b) . ...)
        return connective(
            comparison(x, a),
            OrValue([
                comparison(x, b),
                # Recursive application along associative chains or termination
                # in some other connective or constant.
                connective(
                    comparison(x, b),
                    y
                )
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
        ReorderTernaryComparison_v1
    ]

    exhaustive = True
