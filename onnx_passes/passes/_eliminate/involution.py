from onnx_passes.passes._base import (
    RewriteRuleSetTemplate, RewriteRule, Transformation, Sequential
)
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


class EliminateInvolutionXor_v1(RewriteRule, Verify):
    """Eliminate involution of the boolean Xor operation."""

    @staticmethod
    def pattern(op, x, y):
        return op.Xor(op.Xor(x, y), x)

    @staticmethod
    def rewrite(op, x, y):
        return op.Expand(
            y,
            op.Shape(op.Xor(x, y))
        )

    @property
    def commute(self) -> bool:
        return True


class EliminateInvolutionBitwiseXor_v1(RewriteRule, Verify):
    """Eliminate involution of the bitwise Xor operation."""

    @staticmethod
    def pattern(op, x, y):
        return op.BitwiseXor(op.BitwiseXor(x, y), x)

    @staticmethod
    def rewrite(op, x, y):
        return op.Expand(
            y,
            op.Shape(op.BitwiseXor(x, y))
        )

    @property
    def commute(self) -> bool:
        return True


class EliminateInvolutionLoop_v1(Sequential, Transformation):
    """Exhaustively apply involution elimination transformations."""

    passes = [
        EliminateInvolution_v1,
        EliminateInvolutionXor_v1,
        EliminateInvolutionBitwiseXor_v1
    ]

    exhaustive = True
