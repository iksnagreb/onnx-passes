from onnx_passes.passes.base import RewriteRule
from onnx_passes.passes.verify import Verify, tolerance


@tolerance
class InlineLogSoftmax_v1(RewriteRule, Verify):
    """Inline LogSoftmax operations as Log-Softmax."""

    @staticmethod
    def pattern(op, x, axis):
        return op.LogSoftmax(x, axis=axis)

    @staticmethod
    def rewrite(op, x, axis):
        return op.Log(op.Softmax(x, axis=axis))


@tolerance
class LowerSoftmaxToReduce_v1(RewriteRule, Verify):
    """Lower Softmax operations to Exp-ReduceSum."""

    @staticmethod
    def pattern_v13(op, x, axis):
        return op.Softmax(x, axis=axis)

    @staticmethod
    def pattern_v11(op, x, axis):
        return op.Softmax(x, axis=axis)

    @staticmethod
    def rewrite_v13(op, x, axis):
        return op.Div(
            op.Exp(x),
            op.ReduceSum(
                op.Exp(x),
                op.Reshape(
                    op.Constant(value_int=axis.as_int()),
                    op.Constant(value_ints=[1])
                ),
                keepdims=1
            )
        )

    @staticmethod
    def rewrite_v11(op, x, axis):
        # This should be lowering according to the ONNX specification, however,
        # the reference evaluator is out-of-spec for opset below v13:
        #   https://github.com/onnx/onnx/pull/8607
        return op.Reshape(
            op.Div(
                op.Exp(
                    op.Flatten(
                        x, axis=axis
                    )
                ),
                op.ReduceSum(
                    op.Exp(
                        op.Flatten(
                            x, axis=axis
                        )
                    ),
                    axes=[1],
                    keepdims=1
                )
            ),
            op.Shape(x)
        )
