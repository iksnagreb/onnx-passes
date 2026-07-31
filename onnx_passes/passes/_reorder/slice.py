from onnx_passes.passes._base import RewriteRule
from onnx_passes.passes._verify import Verify

from onnx_passes.traits.elementwise import produced_by_elementwise

import numpy as np
import onnx_ir as ir


class MoveElementwisePastSlice_v1(RewriteRule, Verify):
    """Reorder elementwise operations to follow slicing where applicable."""

    @staticmethod
    def pattern(op, starts, ends, axes, steps):
        return op.Slice(
            produced_by_elementwise, starts, ends, axes, steps, _outputs=["out"]
        )

    @staticmethod
    def rewrite(op, starts, ends, axes, steps, out):
        # Find the elementwise operator which produces the input to the matched
        # reshape operator (the value level check guarantees this exists and is
        # indeed the node we are interested in).
        elementwise = out.producer().inputs[0].producer()

        # Collect the list of inputs to the elementwise operation with all
        # inputs expanded and sliced to match the output shape.
        inputs = []

        for inp in elementwise.inputs:
            inputs.append(
                op.Slice(
                    op.Expand(
                        inp,
                        op.Shape(elementwise.outputs[0])
                    ),
                    starts,
                    ends,
                    axes,
                    steps
                )
            )

        # Insert the replacement pattern with attributes transplanted from the
        # elementwise operator
        return op.op(elementwise.op_type, *inputs, **elementwise.attributes)
