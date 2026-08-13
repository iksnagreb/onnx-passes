from onnx_passes.passes._base import (
    RewriteRuleSetTemplate, Transformation, Sequential
)
from onnx_passes.passes._verify import Verify


class EliminateIdempotence_v1(RewriteRuleSetTemplate, Verify):
    """Eliminate idempotent functions, i.e., repeating has no effect."""

    patterns = (
        lambda op: op.Abs,
        lambda op: op.Ceil,
        lambda op: op.Floor,
        lambda op: op.Round,
        lambda op: op.Relu,
        lambda op: op.Mean,  # Note: Mean(Mean(x)) = Mean(x)
        lambda op: op.Sign
    )

    @staticmethod
    def pattern(partial, op, x):
        return partial(op)(partial(op)(x))

    @staticmethod
    def rewrite(partial, op, x):
        return partial(op)(x)


class EliminateBinaryIdempotence_v1(RewriteRuleSetTemplate, Verify):
    """Eliminate idempotent binary operations, i.e., repeating has no effect."""

    patterns = (
        lambda op: op.And,
        lambda op: op.Or,
        lambda op: op.BitwiseAnd,
        lambda op: op.BitwiseOr,
        lambda op: op.Min,
        lambda op: op.Max,
        lambda op: op.Mean,  # Note: Mean(x, x) = x
    )

    @staticmethod
    def pattern(partial, op, x):
        return partial(op)(x, x)

    @staticmethod
    def rewrite(partial, op, x):
        return op.Identity(x)


class EliminateIdempotenceLoop_v1(Sequential, Transformation):
    """Exhaustively apply idempotence elimination transformations."""

    passes = [
        EliminateIdempotence_v1,
        EliminateBinaryIdempotence_v1
    ]

    exhaustive = True
