from onnx_passes.passes._base import RewriteRule, Transformation, Sequential
from onnx_passes.passes._verify import Verify

from onnx_passes.traits.elementwise import produced_by_elementwise


class MoveElementwisePastExpand_v1(RewriteRule, Verify):
    """Reorder elementwise operations to follow expanding where applicable."""

    @staticmethod
    def pattern(op, shape):
        return op.Expand(produced_by_elementwise, shape, _outputs=["out"])

    @staticmethod
    def rewrite(op, shape, out):
        # Find the elementwise operator which produces the input to the matched
        # expand operator (the value level check guarantees this exists and is
        # indeed the node we are interested in).
        elementwise = out.producer().inputs[0].producer()

        # Collect the list of inputs to the elementwise operation with all
        # inputs expanded and reshaped to match the output shape.
        inputs = []

        for inp in elementwise.inputs:
            inputs.append(op.Expand(inp, shape))

        # Insert the replacement pattern with attributes transplanted from the
        # elementwise operator
        return op.op(elementwise.op_type, *inputs, **elementwise.attributes)


class ReorderExpandLoop_v1(Sequential, Transformation):
    """Exhaustively apply expand reordering transformations."""

    passes = [
        MoveElementwisePastExpand_v1,
    ]

    exhaustive = True
