from onnx_passes.passes._base import RewriteRule, Transformation, Sequential
from onnx_passes.passes._verify import Verify

from onnxscript.rewriter.pattern import OrValue

import onnx_ir as ir
import numpy as np


class EliminateComplementAnd_v1(RewriteRule, Verify):
    """Eliminate complementation of boolean And."""

    @staticmethod
    def pattern(op, x):
        return op.And(x, op.Not(x))

    @staticmethod
    def rewrite(op, x):
        return op.Expand(
            op.CastLike(
                op.Constant(value=ir.tensor(False)),
                x
            ),
            op.Shape(
                x
            )
        )


class EliminateComplementOr_v1(RewriteRule, Verify):
    """Eliminate complementation of boolean Or."""

    @staticmethod
    def pattern(op, x):
        return op.Or(x, op.Not(x))

    @staticmethod
    def rewrite(op, x):
        return op.Expand(
            op.CastLike(
                op.Constant(value=ir.tensor(True)),
                x
            ),
            op.Shape(
                x
            )
        )

    @property
    def commute(self) -> bool:
        return True


class EliminateComplementXor_v1(RewriteRule, Verify):
    """Eliminate complementation of boolean Xor."""

    @staticmethod
    def pattern(op, x):
        return op.Xor(x, OrValue([x, op.Not(x)], tag_var="tag"))

    @staticmethod
    def rewrite(op, x, tag):
        return op.Expand(
            op.CastLike(
                op.Constant(value_int=tag),
                x
            ),
            op.Shape(
                x
            )
        )

    @property
    def commute(self) -> bool:
        return True


class EliminateComplementBitwiseAnd_v1(RewriteRule, Verify):
    """Eliminate complementation of bitwise And."""

    @staticmethod
    def pattern(op, x):
        return op.BitwiseAnd(x, op.BitwiseNot(x))

    @staticmethod
    def rewrite(op, x):
        return op.Expand(
            op.CastLike(
                op.Constant(value_int=0),
                x
            ),
            op.Shape(
                x
            )
        )

    @property
    def commute(self) -> bool:
        return True


class EliminateComplementBitwiseOr_v1(RewriteRule, Verify):
    """Eliminate complementation of bitwise Or."""

    @staticmethod
    def pattern(op, x):
        return op.BitwiseOr(x, op.BitwiseNot(x))

    @staticmethod
    def rewrite(op, x):
        return op.Expand(
            op.CastLike(
                op.Constant(value_int=~0),  # ~ 111...1
                x
            ),
            op.Shape(
                x
            )
        )

    @property
    def commute(self) -> bool:
        return True


class EliminateComplementBitwiseXor_v1(RewriteRule, Verify):
    """Eliminate complementation of bitwise Xor."""

    @staticmethod
    def pattern(op, x):
        return op.BitwiseXor(
            x, OrValue(
                [x, op.BitwiseNot(x)], tag_var="tag", tag_values=[0, ~0]
            )
        )

    @staticmethod
    def rewrite(op, x, tag):
        return op.Expand(
            op.CastLike(
                op.Constant(value_int=tag),
                x
            ),
            op.Shape(
                x
            )
        )

    @property
    def commute(self) -> bool:
        return True


class EliminateComplementTernaryXor_v1(RewriteRule, Verify):
    """Eliminate boolean Xor of constant complementary And.

    Note: This shortcuts Distributive->Constant Folding->Identity Elimination
    """

    @staticmethod
    def pattern(op, a, b, x):
        return op.Xor(op.And(a, x), op.And(b, x))

    @staticmethod
    def check(context, a, b, x):
        if (a := ir.convenience.get_const_tensor(a)) is not None:
            if (b := ir.convenience.get_const_tensor(b)) is not None:
                return np.all(a.numpy() != b.numpy())
        return False

    @staticmethod
    def rewrite(op, a, b, x):
        return op.Identity(x)

    @property
    def commute(self) -> bool:
        return True


class EliminateComplementTernaryBitwiseXor_v1(RewriteRule, Verify):
    """Eliminate bitwise Xor of constant complementary And.

    Note: This shortcuts Distributive->Constant Folding->Identity Elimination
    """

    @staticmethod
    def pattern(op, a, b, x):
        return op.BitwiseXor(op.BitwiseAnd(a, x), op.BitwiseAnd(b, x))

    @staticmethod
    def check(context, a, b, x):
        if (a := ir.convenience.get_const_tensor(a)) is not None:
            if (b := ir.convenience.get_const_tensor(b)) is not None:
                return np.all(a.numpy() == ~b.numpy())
        return False

    @staticmethod
    def rewrite(op, a, b, x):
        return op.Identity(x)

    @property
    def commute(self) -> bool:
        return True


class EliminateComplementLoop_v1(Sequential, Transformation):
    """Exhaustively apply complement elimination transformations."""

    passes = [
        EliminateComplementAnd_v1,
        EliminateComplementOr_v1,
        EliminateComplementXor_v1,
        EliminateComplementBitwiseAnd_v1,
        EliminateComplementBitwiseOr_v1,
        EliminateComplementBitwiseXor_v1,
        EliminateComplementTernaryXor_v1,
        EliminateComplementTernaryBitwiseXor_v1
    ]

    exhaustive = True
