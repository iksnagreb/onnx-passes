from onnx_passes.passes._base import RewriteRuleSetTemplate
from onnx_passes.passes._verify import Verify

import onnx_ir as ir


class AbsorbAbsIntoComparison_v1(RewriteRuleSetTemplate, Verify):
    """Rewrite comparisons to the absolute value function into constant rhs.

    Derived by inlining |x| -> x >= 0 ? +x : -x into the (in-)equality and
    flattening as a boolean expression similar RewriteBooleanWhereAsOr_v1.
    """

    patterns = (
        lambda op: op.Greater,
        lambda op: op.Less,
        lambda op: op.Equal,
        lambda op: op.GreaterOrEqual,
        lambda op: op.LessOrEqual,
    )

    @staticmethod
    def pattern(partial, op, x, c):
        return partial(op)(op.Abs(x), c)

    @staticmethod
    def check(context, x: ir.Value, c):
        if x.dtype.is_signed():
            return ir.convenience.get_const_tensor(c) is not None

        return False

    @staticmethod
    def rewrite(partial, op, x, c):
        return op.Or(
            op.And(
                op.GreaterOrEqual(
                    x,
                    op.CastLike(
                        op.Constant(value_float=0.0),
                        x
                    )
                ),
                partial(op)(x, c)
            ),
            op.And(
                op.LessOrEqual(
                    x,
                    op.CastLike(
                        op.Constant(value_float=0.0),
                        x
                    )
                ),
                partial(op)(
                    op.Neg(x),
                    c
                )
            )
        )
