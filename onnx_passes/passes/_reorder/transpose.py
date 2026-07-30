from onnx_passes.passes._base import RewriteRule
from onnx_passes.passes._verify import Verify

from onnx_passes.traits.elementwise import produced_by_elementwise


class MoveElementwisePastTranspose_v1(RewriteRule, Verify):
    """Reorder elementwise operations to follow transpose where applicable."""

    @staticmethod
    def pattern(op, perm):
        return op.Transpose(
            produced_by_elementwise, perm=perm, _outputs=["out"]
        )

    @staticmethod
    def rewrite(op, perm, out):
        # Find the elementwise operator which produces the input to the matched
        # transpose operator (the value level check guarantees this exists and
        # is indeed the node we are interested in).
        elementwise = out.producer().inputs[0].producer()

        # Collect the list of inputs to the elementwise operation with all
        # inputs expanded and transposed to match the output shape.
        inputs = []

        for inp in elementwise.inputs:
            inputs.append(
                op.Transpose(
                    op.Expand(
                        inp,
                        op.Shape(elementwise.outputs[0])
                    ),
                    perm=perm
                )
            )

        # Insert the replacement pattern with attributes transplanted from the
        # elementwise operator
        return op.op(
            elementwise.op_type, inputs, attributes=elementwise.attributes
        )
