from onnx_passes.passes._base import RewriteRuleSetTemplate
from onnx_passes.passes._verify import Verify

import onnx_ir as ir


class AbsorbReciprocalIntoComparison_v1(RewriteRuleSetTemplate, Verify):
    """Rewrite comparisons to absorb the Reciprocal function.

    This effectively solves inequalities 1 / x > c for all three cases of x. For
    equalities, 2. and 3. end up identical/redundant.
        1. x = 0 -> 0 > c
        2. a > 0 -> x > c / a
        3. a < 0 -> x < c / a
    """

    patterns = (
        lambda op: op.GreaterOrEqual,
        lambda op: op.LessOrEqual,
        lambda op: op.Greater,
        lambda op: op.Less,
        lambda op: op.Equal
    )

    @staticmethod
    def pattern(partial, op, x, c):
        return partial(op)(op.Reciprocal(x), c)

    @staticmethod
    def check(context, x, c):
        return ir.convenience.get_const_tensor(c) is not None

    @staticmethod
    def rewrite(partial, op, x, c):
        return op.Or(
            # x >= 0 & 1 >= c * x
            op.And(
                op.GreaterOrEqual(
                    x,
                    op.CastLike(
                        op.Constant(value_float=0.0),
                        x
                    )
                ),
                partial(op)(
                    op.CastLike(
                        op.Constant(value_float=1.0),
                        x
                    ),
                    op.Mul(
                        c,
                        x
                    )
                )
            ),
            # x <= 0 & 1 <= c * x -> x <= 0 & c * x >= 1
            op.And(
                op.LessOrEqual(
                    x,
                    op.CastLike(
                        op.Constant(value_float=0.0),
                        x
                    )
                ),
                partial(op)(
                    op.Mul(
                        c,
                        x
                    ),
                    op.CastLike(
                        op.Constant(value_float=1.0),
                        x
                    )
                )
            )
        )
