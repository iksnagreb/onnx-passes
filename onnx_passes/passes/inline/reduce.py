from onnx_passes.passes.base import RewriteRule
from onnx_passes.passes.verify import Verify, tolerance

import onnx_ir as ir


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
        # Collect original attributes (ReduceMean and ReduceSum share supported
        # attributes and defaults) and infer the default noop_with_empty_axes.
        attributes = out.producer().attributes

        if not (noop_with_empty_axes := attributes.get("noop_with_empty_axes")):
            noop_with_empty_axes = ir.Attr(
                "noop_with_empty_axes", ir.AttributeType.INT, 0
            )

        noop_with_empty_axes = noop_with_empty_axes.as_int()

        return op.Div(
            op.ReduceSum(
                x, axes, **out.producer().attributes
            ),
            op.Squeeze(
                op.CastLike(
                    op.Where(
                        op.Equal(
                            op.Constant(value_int=noop_with_empty_axes),
                            op.Constant(value_int=1)
                        ),
                        # Noop with empty axes, i.e., empty axes does not reduce
                        # and the scale is 1 (empty product is 1).
                        scale := op.ReduceProd(
                            op.Gather(
                                op.Shape(x),
                                axes
                            )
                        ),
                        # Not noop with empty axes, i.e., empty axes reduce over
                        # all axes
                        op.Where(
                            op.Equal(
                                op.Size(axes),
                                op.Constant(value_int=0)
                            ),
                            op.ReduceProd(
                                op.Shape(x)
                            ),
                            scale
                        )
                    ),
                    x
                )
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
