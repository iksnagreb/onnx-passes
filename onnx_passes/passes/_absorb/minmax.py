from onnx_passes.passes._base import (
    RewriteRuleSetTemplate, Transformation, Sequential
)
from onnx_passes.passes._verify import Verify

import onnx_ir as ir


class AbsorbMinIntoComparison_v1(RewriteRuleSetTemplate, Verify):
    """Rewrite comparisons to absorb constant minimum into constant right.

    This effectively solves (in-)equalities by inlining the min operation into a
    boolean or expression: min(x, a) ? c -> x <= a & x ? c | a <= x & a ? c
    """

    patterns = (
        lambda op: op.Equal,
        lambda op: op.Greater,
        lambda op: op.GreaterOrEqual,
        lambda op: op.Less,
        lambda op: op.LessOrEqual,
    )

    @staticmethod
    def pattern(partial, op, x, a, c):
        return partial(op)(op.Min(x, a), c)

    @staticmethod
    def check(context, x, a, c):
        if ir.convenience.get_const_tensor(a) is None:
            return False

        if ir.convenience.get_const_tensor(c) is None:
            return False

        return True

    @staticmethod
    def rewrite(partial, op, x, a, c):
        return op.Or(
            op.And(
                op.LessOrEqual(x, a),
                partial(op)(x, c)
            ),
            op.And(
                op.LessOrEqual(a, x),
                partial(op)(a, c)
            )
        )


class AbsorbMaxIntoComparison_v1(RewriteRuleSetTemplate, Verify):
    """Rewrite comparisons to absorb constant maximum into constant right.

    This effectively solves (in-)equalities by inlining the max operation into a
    boolean or expression: max(x, a) ? c -> x >= a & x ? c | a >= x & a ? c
    """

    patterns = (
        lambda op: op.Equal,
        lambda op: op.Greater,
        lambda op: op.GreaterOrEqual,
        lambda op: op.Less,
        lambda op: op.LessOrEqual,
    )

    @staticmethod
    def pattern(partial, op, x, a, c):
        return partial(op)(op.Max(x, a), c)

    @staticmethod
    def check(context, x, a, c):
        if ir.convenience.get_const_tensor(a) is None:
            return False

        if ir.convenience.get_const_tensor(c) is None:
            return False

        return True

    @staticmethod
    def rewrite(partial, op, x, a, c):
        return op.Or(
            op.And(
                op.GreaterOrEqual(x, a),
                partial(op)(x, c)
            ),
            op.And(
                op.GreaterOrEqual(a, x),
                partial(op)(a, c)
            )
        )


class AbsorbMinMaxLoop_v1(Sequential, Transformation):
    """Exhaustively apply min/max absorption transformations."""

    passes = [
        AbsorbMinIntoComparison_v1,
        AbsorbMaxIntoComparison_v1,
    ]

    exhaustive = True
