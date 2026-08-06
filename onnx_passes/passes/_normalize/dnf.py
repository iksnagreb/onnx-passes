from onnx_passes.passes._base import RewriteRuleSet, Sequential, Transformation
from onnx_passes.passes._verify import Verify


class RewriteXorAsPrimitiveDNF_v1(RewriteRuleSet, Verify):
    """Rewrites boolean/bitwise Xor as primitives in disjunctive normal form."""

    @property
    def commute(self) -> bool:
        return True

    @staticmethod
    def pattern():
        return [
            lambda op, x, y: op.Xor(x, y),
            lambda op, x, y: op.BitwiseXor(x, y),
        ]

    @staticmethod
    def rewrite():
        return [
            lambda op, x, y: op.Or(
                op.And(op.Not(x), y),
                op.And(x, op.Not(y))
            ),
            lambda op, x, y: op.BitwiseOr(
                op.Bitwisend(op.BitwiseNot(x), y),
                op.BitwiseAnd(x, op.BitwiseNot(y))
            )
        ]


class PrimitiveBooleanToDNF_v1(RewriteRuleSet, Verify):
    """Converts primitive boolean expressions to disjunctive normal form."""

    @property
    def commute(self) -> bool:
        return True

    @staticmethod
    def pattern():
        return [
            lambda op, x: op.Not(op.Not(x)),
            lambda op, x, y: op.Not(op.Or(x, y)),
            lambda op, x, y: op.Not(op.And(x, y)),
            lambda op, x, y, z: op.And(x, op.Or(y, z)),
            lambda op, x, y, z: op.And(op.Or(x, y), z),
        ]

    @staticmethod
    def rewrite():
        return [
            lambda op, x: x,
            lambda op, x, y: op.And(op.Not(x), op.Not(y)),
            lambda op, x, y: op.Or(op.Not(x), op.Not(y)),
            lambda op, x, y, z: op.Or(op.And(x, y), op.And(x, z)),
            lambda op, x, y, z: op.Or(op.And(x, z), op.And(y, z)),
        ]


class PrimitiveBooleanToDNFLoop_v1(Sequential, Transformation):
    """Exhaustively applies the boolean to DNF term rewriting system."""

    passes = [
        PrimitiveBooleanToDNF_v1
    ]

    exhaustive = True


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
