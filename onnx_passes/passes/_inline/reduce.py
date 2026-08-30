from onnx_passes.passes._base import RewriteRule
from onnx_passes.passes._verify import Verify, tolerance


class InlineReduceL1_v1(RewriteRule, Verify):
    """Inline ReduceL1 reduction as Abs-ReduceSum."""

    @staticmethod
    def pattern_v18(op, x, axes):
        return op.ReduceL1(x, axes, _outputs=["out"])

    @staticmethod
    def rewrite_v18(op, x, axes, out):
        return op.ReduceSum(op.Abs(x), axes, **out.producer().attributes)


@tolerance
class InlineReduceL2_v1(RewriteRule, Verify):
    """Inline ReduceL2 reduction as Mul-ReduceSum-Sqrt."""

    @staticmethod
    def pattern_v18(op, x, axes):
        return op.ReduceL2(x, axes, _outputs=["out"])

    @staticmethod
    def rewrite_v18(op, x, axes, out):
        return op.Sqrt(
            op.ReduceSum(op.Mul(x, x), axes, **out.producer().attributes)
        )


class InlineReduceLogSum_v1(RewriteRule, Verify):
    """Inline ReduceLogSum reduction as ReduceSum-Log."""

    @staticmethod
    def pattern_v18(op, x, axes):
        return op.ReduceLogSum(x, axes, _outputs=["out"])

    @staticmethod
    def rewrite_v18(op, x, axes, out):
        return op.Log(op.ReduceSum(x, axes, **out.producer().attributes))


@tolerance
class InlineReduceLogSumExp_v1(RewriteRule, Verify):
    """Inline ReduceLogSumExp reduction as Exp-ReduceSum-Log."""

    @staticmethod
    def pattern_v18(op, x, axes):
        return op.ReduceLogSumExp(x, axes, _outputs=["out"])

    @staticmethod
    def rewrite_v18(op, x, axes, out):
        return op.Log(
            op.ReduceSum(op.Exp(x), axes, **out.producer().attributes)
        )


@tolerance
class InlineReduceMean_v1(RewriteRule, Verify):
    """Inline ReduceMean reduction as ReduceSum-Div."""

    @staticmethod
    def pattern_v18(op, x, axes):
        return op.ReduceMean(x, axes, _outputs=["out"])

    @staticmethod
    def rewrite_v18(op, x, axes, out):
        return op.Div(
            op.ReduceSum(
                x, axes, **out.producer().attributes
            ),
            op.CastLike(
                op.ReduceProd(
                    op.Gather(op.Shape(x), axes)
                ),
                x
            )
        )


class InlineReduceSumSquare_v1(RewriteRule, Verify):
    """Inline ReduceSumSquare reduction as Mul-ReduceSum."""

    @staticmethod
    def pattern_v18(op, x, axes):
        return op.ReduceSumSquare(x, axes, _outputs=["out"])

    @staticmethod
    def rewrite_v18(op, x, axes, out):
        return op.ReduceSum(op.Mul(x, x), axes, **out.producer().attributes)
