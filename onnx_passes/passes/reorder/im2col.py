from onnx_passes.passes.base import RewriteRule
from onnx_passes.passes.verify import Verify

from onnx_passes.traits.elementwise import produced_by_elementwise

from onnx_passes.ops import DOMAIN as CUSTOM_DOMAIN

import onnx_ir as ir


class MoveElementwisePastIm2Col_v1(RewriteRule, Verify):
    """Reorder elementwise operations to follow Im2Col where applicable."""

    @staticmethod
    def pattern(op):
        return op.Im2Col(
            produced_by_elementwise, _outputs=["out"], _domain=CUSTOM_DOMAIN
        )

    @staticmethod
    def check(context, out):
        # As input generators tend to be expensive, do not reorder this if there
        # is more than one non-constant input.
        xs = out.producer().inputs[0].producer().inputs
        return sum(ir.convenience.get_const_tensor(x) is None for x in xs) == 1

    @staticmethod
    def rewrite(op, out):
        # Find the elementwise operator which produces the input to the matched
        # Im2Col operator (the value level check guarantees this exists and is
        # indeed the node we are interested in).
        elementwise = out.producer().inputs[0].producer()

        # Collect the list of inputs to the elementwise operation with all
        # inputs expanded before generating sliding windows.
        inputs = []

        for inp in elementwise.inputs:
            inputs.append(
                op.Im2Col(
                    op.Expand(
                        inp,
                        op.Shape(
                            op.op(
                                elementwise.op_type,
                                *elementwise.inputs,
                                **elementwise.attributes
                            )
                        )
                    ),
                    **out.producer().attributes,
                    _domain=CUSTOM_DOMAIN,
                    _version=out.producer().version
                )
            )

        # Insert the replacement pattern with attributes transplanted from the
        # elementwise operator
        return op.op(elementwise.op_type, *inputs, **elementwise.attributes)
