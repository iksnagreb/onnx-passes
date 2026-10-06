from onnx_passes.passes._base import RewriteRuleSetTemplate
from onnx_passes.passes._verify import Verify

import onnx_ir as ir


class AbsorbLogIntoComparison_v1(RewriteRuleSetTemplate, Verify):
    """Rewrite comparisons to absorb logarithm into constant rhs."""

    patterns = (
        lambda op: op.Greater,
        lambda op: op.Less,
        lambda op: op.Equal,
        lambda op: op.GreaterOrEqual,
        lambda op: op.LessOrEqual,
    )

    @staticmethod
    def pattern(partial, op, x, c):
        return partial(op)(op.Log(x), c)

    @staticmethod
    def check(context, x, c):
        return ir.convenience.get_const_tensor(c) is not None

    @staticmethod
    def rewrite(partial, op, x, c):
        return op.And(
            op.GreaterOrEqual(
                x,
                op.CastLike(
                    op.Constant(value_float=0.0),
                    x
                )
            ),
            partial(op)(x, op.Exp(c))
        )
