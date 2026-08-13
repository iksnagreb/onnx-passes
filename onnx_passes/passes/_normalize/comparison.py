from onnx_passes.passes._base import RewriteRuleSet
from onnx_passes.passes._verify import Verify


class RewriteLessAsGreater_v1(RewriteRuleSet, Verify):
    """Rewrite Less than comparison as Greater than comparison."""

    @staticmethod
    def pattern():
        return [
            lambda op, x, y: op.Less(x, y),
            lambda op, x, y: op.LessOrEqual(x, y),
        ]

    @staticmethod
    def rewrite():
        return [
            lambda op, x, y: op.Greater(y, x),
            lambda op, x, y: op.GreaterOrEqual(y, x),
        ]


class SimplifyCompoundComparison_v1(RewriteRuleSet, Verify):
    """Simplify compound (And/Or) comparison to a common variable."""

    @staticmethod
    def pattern():
        return [
            # Common variable x on the left side
            lambda op, x, a, b: op.And(
                op.Greater(x, a), op.Greater(x, b)
            ),
            lambda op, x, a, b: op.And(
                op.GreaterOrEqual(x, a), op.GreaterOrEqual(x, b)
            ),
            lambda op, x, a, b: op.Or(
                op.Greater(x, a), op.Greater(x, b)
            ),
            lambda op, x, a, b: op.Or(
                op.GreaterOrEqual(x, a), op.GreaterOrEqual(x, b)
            ),
            # Common variable x on the right side
            lambda op, x, a, b: op.And(
                op.Greater(a, x), op.Greater(b, x)
            ),
            lambda op, x, a, b: op.And(
                op.GreaterOrEqual(a, x), op.GreaterOrEqual(b, x)
            ),
            lambda op, x, a, b: op.Or(
                op.Greater(a, x), op.Greater(b, x)
            ),
            lambda op, x, a, b: op.Or(
                op.GreaterOrEqual(a, x), op.GreaterOrEqual(b, x)
            )
        ]

    @staticmethod
    def rewrite():
        return [
            # Common variable x on the left side
            lambda op, x, a, b: op.Greater(x, op.Max(a, b)),
            lambda op, x, a, b: op.GreaterOrEqual(x, op.Max(a, b)),
            lambda op, x, a, b: op.Greater(x, op.Min(a, b)),
            lambda op, x, a, b: op.GreaterOrEqual(x, op.Min(a, b)),
            # Common variable x on the right side
            lambda op, x, a, b: op.Greater(op.Min(a, b), x),
            lambda op, x, a, b: op.GreaterOrEqual(op.Min(a, b), x),
            lambda op, x, a, b: op.Greater(op.Max(a, b), x),
            lambda op, x, a, b: op.GreaterOrEqual(op.Max(a, b), x)
        ]
