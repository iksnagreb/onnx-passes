from onnx_passes.passes._base import RewriteRuleSet, Sequential, Transformation
from onnx_passes.passes._verify import Verify


class PrimitiveBitwiseToDNF_v1(RewriteRuleSet, Verify):
    """Converts primitive bitwise expressions to disjunctive normal form."""

    @property
    def commute(self) -> bool:
        return True

    @staticmethod
    def pattern():
        return [
            lambda op, x: op.BitwiseNot(op.BitwiseNot(x)),
            lambda op, x, y: op.BitwiseNot(op.BitwiseOr(x, y)),
            lambda op, x, y: op.BitwiseNot(op.BitwiseAnd(x, y)),
            lambda op, x, y, z: op.BitwiseAnd(x, op.BitwiseOr(y, z)),
            lambda op, x, y, z: op.BitwiseAnd(op.BitwiseOr(x, y), z),
        ]

    @staticmethod
    def rewrite():
        return [
            lambda op, x: x,
            lambda op, x, y: \
                op.BitwiseAnd(op.BitwiseNot(x), op.BitwiseNot(y)),
            lambda op, x, y: \
                op.BitwiseOr(op.BitwiseNot(x), op.BitwiseNot(y)),
            lambda op, x, y, z: \
                op.BitwiseOr(op.BitwiseAnd(x, y), op.BitwiseAnd(x, z)),
            lambda op, x, y, z: \
                op.BitwiseOr(op.BitwiseAnd(x, z), op.BitwiseAnd(y, z)),
        ]


class PrimitiveBitwiseToDNFLoop_v1(Sequential, Transformation):
    """Exhaustively applies the bitwise to DNF term rewriting system."""

    passes = [
        PrimitiveBitwiseToDNF_v1
    ]

    exhaustive = True
