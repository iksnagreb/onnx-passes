from onnx_passes.passes._base import RewriteRule
from onnx_passes.passes._verify import Verify

from onnxscript.rewriter.pattern import OrValue

import onnx_ir as ir


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
