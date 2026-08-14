from onnx_passes.passes._base import RewriteRuleSet, Transformation, Sequential
from onnx_passes.passes._verify import Verify

import onnx_ir as ir


class BooleanToANF_v1(RewriteRuleSet, Verify):
    """Convert boolean expressions to algebraic normal form."""

    @staticmethod
    def pattern():
        return [
            lambda op, x: op.Not(x),
            lambda op, x, y: op.Or(x, y),
            lambda op, x, y, z: op.And(x, op.Xor(y, z))
        ]

    @staticmethod
    def rewrite():
        return [
            lambda op, x: op.Xor(op.Constant(value=ir.tensor(True)), x),
            lambda op, x, y: op.Xor(x, op.Xor(y, op.And(x, y))),
            lambda op, x, y, z: op.Xor(op.And(x, y), op.And(x, z))
        ]

    @property
    def commute(self) -> bool:
        return True


class BitwiseToANF_v1(RewriteRuleSet, Verify):
    """Convert bitwise expressions to algebraic normal form."""

    @staticmethod
    def pattern():
        return [
            lambda op, x: op.BitwiseNot(x),
            lambda op, x, y: op.BitwiseOr(x, y),
            lambda op, x, y, z: op.BitwiseAnd(x, op.BitwiseXor(y, z))
        ]

    @staticmethod
    def rewrite():
        return [
            lambda op, x: op.BitwiseXor(
                op.CastLike(op.Constant(value=ir.tensor(1)), x), x
            ),
            lambda op, x, y: op.BitwiseXor(
                x, op.BitwiseXor(y, op.BitwiseAnd(x, y))
            ),
            lambda op, x, y, z: op.BitwiseXor(
                op.BitwiseAnd(x, y), op.BitwiseAnd(x, z)
            )
        ]

    @property
    def commute(self) -> bool:
        return True


class BooleanToANFLoop_v1(Sequential, Transformation):
    """Exhaustively apply the boolean to ANF term rewriting system."""

    passes = [
        BooleanToANF_v1,
    ]

    exhaustive = True


class BitwiseToANFLoop_v1(Sequential, Transformation):
    """Exhaustively apply the bitwise to ANF term rewriting system."""

    passes = [
        BitwiseToANF_v1,
    ]

    exhaustive = True
