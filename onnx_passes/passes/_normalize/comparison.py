from onnx_passes.passes._base import RewriteRuleSet, Sequential, Transformation
from onnx_passes.passes._verify import Verify


class RewriteLessAsGreater_v1(RewriteRuleSet, Verify):
    """Rewrites Less than comparison as Greater than comparison."""

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
