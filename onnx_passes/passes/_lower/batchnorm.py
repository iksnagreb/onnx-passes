from onnx_passes.passes._base import RewriteRule, Sequential
from onnx_passes.passes._verify import Verify, tolerance


@tolerance
class LowerBatchNormalizationToMulAdd_v1(RewriteRule, Verify):
    """Lower BatchNormalization to Mul and Add along the channel dimension.

    This only applies to inference mode BatchNormalization with a single output
    tensor, i.e., running mean/variance updates are not lowered.
    """

    @staticmethod
    def pattern_v9(op, x, scale, b, mean, var, epsilon):
        return op.BatchNormalization(x, scale, b, mean, var, epsilon=epsilon)

    @staticmethod
    def check_v9(context, x, scale, b, mean, var, epsilon):
        # Decide training vs. inference mode based on the presence of optional
        # mean/variance outputs
        return len(context.root.outputs) == 1

    @staticmethod
    def rewrite_v9(op, x, scale, b, mean, var, epsilon):
        return op.Add(
            op.Mul(
                op.Div(
                    op.Sub(
                        x,
                        op.Reshape(
                            mean,
                            # Expand matching x1x1x... spatial dimensions at the
                            # end of the batchnorm parameter tensors
                            expanded_shape := op.Concat(
                                op.Shape(
                                    x, start=1, end=2
                                ),
                                op.Expand(
                                    op.Constant(value_ints=[1]),
                                    op.Shape(
                                        op.Shape(x, start=2)
                                    )
                                ),
                                axis=0
                            )
                        )
                    ),
                    op.Sqrt(
                        op.Add(
                            op.Reshape(
                                var,
                                expanded_shape
                            ),
                            op.CastLike(
                                op.Constant(
                                    value_float=epsilon.as_float()
                                ),
                                var
                            )
                        )
                    )
                ),
                op.Reshape(
                    scale,
                    expanded_shape
                )
            ),
            op.Reshape(
                b,
                expanded_shape
            )
        )


# Note: The following is merely a workaround until the potential endless cycle
# between default attributes and unused removal is resolved:
#
#   See https://github.com/onnx/ir-py/issues/507 for details
#
# Still, BatchNormalization lowering must be applied to any model that contains
# at least one inference-mode BatchNormalization before further normalization
# passes or any pass sequence that applies AddDefaultAttributes_v1 followed by
# cleanup (unused removal) exhaustively...
from onnx_passes.passes._normalize.attributes import AddDefaultAttributes_v1


class NormalizeAndLowerBatchNorm_v1(Sequential):
    """Normalize and lower BatchNormalization to Mul and Add."""

    passes = [
        AddDefaultAttributes_v1,
        LowerBatchNormalizationToMulAdd_v1,
    ]
