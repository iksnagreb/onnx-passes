from onnx_passes.passes._base import RewriteRule
from onnx_passes.passes._verify import Verify


class RewriteSignAsWhere_v1(RewriteRule, Verify):
    """Rewrite Sign function as three branches of ternary Where operation."""

    @staticmethod
    def pattern(op, x):
        return op.Sign(x)

    @staticmethod
    def rewrite(op, x):
        return op.Where(
            op.Greater(
                x,
                op.CastLike(
                    op.Constant(value_int=0),
                    x
                )
            ),
            op.CastLike(
                op.Constant(value_int=+1),
                x
            ),
            op.Where(
                op.Less(
                    x,
                    op.CastLike(
                        op.Constant(value_int=0),
                        x
                    )
                ),
                op.CastLike(
                    op.Constant(value_int=-1),
                    x
                ),
                op.CastLike(
                    op.Constant(value_int=+0),
                    x
                )
            )
        )
