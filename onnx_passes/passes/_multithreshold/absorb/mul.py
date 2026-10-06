from onnx_passes.passes._base import RewriteRuleSetTemplate
from onnx_passes.passes._verify import Verify

import onnx_ir as ir


class AbsorbMulIntoComparison_v1(RewriteRuleSetTemplate, Verify):
    """Rewrite comparisons to absorb constant multiplications.

    This effectively solves inequalities a * x > c for all three cases of a. For
    equalities, 2. and 3. end up identical/redundant.
        1. a = 0 -> 0 > c
        2. a > 0 -> x > c / a
        3. a < 0 -> x < c / a
    """

    patterns = (
        lambda op: (op.GreaterOrEqual, op.LessOrEqual),
        lambda op: (op.LessOrEqual, op.GreaterOrEqual),
        lambda op: (op.Greater, op.Less),
        lambda op: (op.Less, op.Greater),
        lambda op: (op.Equal, op.Equal)
    )

    @property
    def commute(self) -> bool:
        return True

    @staticmethod
    def pattern(partial, op, x, a, c):
        return partial(op)[0](op.Mul(x, a), c)

    @staticmethod
    def check(context, x, a, c):
        if ir.convenience.get_const_tensor(a) is None:
            return False

        if ir.convenience.get_const_tensor(c) is None:
            return False

        return True

    @staticmethod
    def rewrite(partial, op, x, a, c):
        return op.Xor(
            # a = 0 & 0 > c
            op.And(
                op.Equal(
                    a,
                    op.CastLike(
                        op.Constant(value_float=0.0), a
                    )
                ),
                partial(op)[0](
                    op.CastLike(
                        op.Constant(value_float=0.0), a
                    ),
                    c
                )
            ),
            op.Xor(
                # a > 0 & x > c / a
                op.And(
                    op.Greater(
                        a,
                        op.CastLike(
                            op.Constant(value_float=0.0), a
                        )
                    ),
                    partial(op)[0](
                        x,
                        op.Div(
                            c,
                            op.Where(
                                op.Equal(
                                    a,
                                    op.CastLike(
                                        op.Constant(value_float=0.0), a
                                    )
                                ),
                                op.CastLike(
                                    op.Constant(value_float=1.0), a
                                ),
                                a
                            )
                        )
                    )
                ),
                # a < 0 & x < c / a
                op.And(
                    op.Less(
                        a,
                        op.CastLike(
                            op.Constant(value_float=0.0), a
                        )
                    ),
                    partial(op)[1](
                        x,
                        op.Div(
                            c,
                            op.Where(
                                op.Equal(
                                    a,
                                    op.CastLike(
                                        op.Constant(value_float=0.0), a
                                    )
                                ),
                                op.CastLike(
                                    op.Constant(value_float=1.0), a
                                ),
                                a
                            )
                        )
                    )
                )
            )
        )
