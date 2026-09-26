from onnx_passes.passes._base import RewriteRule, Transformation, Sequential
from onnx_passes.passes._verify import Verify

from onnx_passes.traits.elementwise import produced_by_elementwise
from onnx_passes.traits.expand import produced_by_expand


class MoveExpandPastElementwise_v1(RewriteRule, Verify):
    """Reorder Expand operations to follow elementwise operations."""

    @staticmethod
    def pattern(op):
        return op.submodule("")(_allow_other_inputs=True, _outputs=["out"])

    @staticmethod
    def check(context, out):
        if produced_by_elementwise(context, out):
            # Accept rewrite as soon as any input is expanded and the overall
            # output shape is static
            for x in out.producer().inputs:
                if produced_by_expand(context, x):
                    return out.shape is not None and out.shape.is_static()

        return False

    @staticmethod
    def rewrite(op, out):
        # Find the elementwise operator which produces the matched value (the
        # value level check guarantees this exists and is indeed the node we are
        # interested in).
        elementwise = out.producer()

        # Collect the list of inputs to the elementwise operation with Expand
        # removed from all inputs.
        inputs = []

        for inp in elementwise.inputs:
            if produced_by_expand(None, inp):
                inp = inp.producer().inputs[0]

            inputs.append(inp)

        # Insert the replacement pattern with attributes transplanted from the
        # elementwise operator and final output expanded
        return op.Expand(
            op.op(
                elementwise.op_type, *inputs, **elementwise.attributes
            ),
            op.Constant(value_ints=out.shape[:])
        )


class ReorderExpandLoop_v1(Sequential, Transformation):
    """Exhaustively apply expand reordering transformations."""

    passes = [
        MoveExpandPastElementwise_v1,
    ]

    exhaustive = True
