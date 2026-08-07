from onnx_passes.passes._base import RewriteRuleSetTemplate
from onnx_passes.passes._verify import Verify


class EliminateInvolution_v1(RewriteRuleSetTemplate, Verify):
    """Eliminate involutions, i.e., self-inverse functions."""

    patterns = (
        lambda op: op.Neg,
        lambda op: op.Not,
        lambda op: op.BitwiseNot,
        lambda op: op.Reciprocal
    )

    @staticmethod
    def pattern(partial, op, x):
        return partial(op)(partial(op)(x))

    @staticmethod
    def rewrite(partial, op, x):
        return op.Identity(x)
