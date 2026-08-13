from onnx_passes.passes._base import RewriteRuleSet, Sequential, Transformation
from onnx_passes.passes._verify import Verify


class RewriteXorAsDNF_v1(RewriteRuleSet, Verify):
    """Rewrite boolean/bitwise Xor as primitives in disjunctive normal form."""

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
                op.BitwiseAnd(op.BitwiseNot(x), y),
                op.BitwiseAnd(x, op.BitwiseNot(y))
            )
        ]


class BooleanToDNF_v1(RewriteRuleSet, Verify):
    """Convert boolean expressions to disjunctive normal form."""

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


class BitwiseToDNF_v1(RewriteRuleSet, Verify):
    """Convert bitwise expressions to disjunctive normal form."""

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


# Reuse common reordering passes to arrive at a more minimal representation, but
# do not use the distributive reordering: here, we always want to distribute and
# operations to end up with a sum of products.
from onnx_passes.passes._reorder import commutative
from onnx_passes.passes._reorder import associative


class BooleanToDNFLoop_v1(Sequential, Transformation):
    """Exhaustively apply the boolean to DNF term rewriting system."""

    passes = [
        BooleanToDNF_v1,
        commutative,
        associative
    ]

    exhaustive = True


class BitwiseToDNFLoop_v1(Sequential, Transformation):
    """Exhaustively apply the bitwise to DNF term rewriting system."""

    passes = [
        BitwiseToDNF_v1,
        commutative,
        associative
    ]

    exhaustive = True
