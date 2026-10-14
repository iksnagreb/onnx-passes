from onnx_passes.passes._base import RewriteRule
from onnx_passes.passes._verify import Verify, tolerance


class InferGemmBias_v1(RewriteRule, Verify):
    """Explicitly infer optional zero bias for Gemm operations."""

    @staticmethod
    def pattern(op, a, b, ):
        return op.Gemm(a, b, _outputs=["out"])

    @staticmethod
    def rewrite(op, a, b, out):
        return op.Gemm(
            a,
            b,
            op.CastLike(
                op.Constant(value_float=0.0),
                a
            ),
            **out.producer().attributes
        )


@tolerance
class LowerGemmToMatMul_v1(RewriteRule, Verify):
    """Lower Gemm operations Transpose-MatMul-Add combination."""

    @staticmethod
    def pattern(op, a, b, c, alpha, beta, trans_a, trans_b):
        return op.Gemm(
            a, b, c, alpha=alpha, beta=beta, transA=trans_a, transB=trans_b
        )

    @staticmethod
    def rewrite(op, a, b, c, alpha, beta, trans_a, trans_b):
        return op.Add(
            op.Mul(
                op.MatMul(
                    [a, op.Transpose(a)][trans_a.as_int()],
                    [b, op.Transpose(b)][trans_b.as_int()],
                ),
                op.CastLike(
                    op.Constant(value_float=alpha.as_float()),
                    a
                )
            ),
            op.Mul(
                c,
                op.CastLike(
                    op.Constant(value_float=beta.as_float()),
                    c
                )
            )
        )
