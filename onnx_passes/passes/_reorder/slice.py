from onnx_passes.passes._base import RewriteRule
from onnx_passes.passes._verify import Verify, tolerance

from onnx_passes.traits.elementwise import produced_by_elementwise

import numpy as np
import onnx_ir as ir


class MoveElementwisePastSlice_v1(RewriteRule, Verify):
    """Reorder elementwise operations to follow slicing where applicable."""

    @staticmethod
    def pattern_v10(op, starts, ends, axes, steps):
        return op.Slice(
            produced_by_elementwise, starts, ends, axes, steps, _outputs=["out"]
        )

    @staticmethod
    def rewrite_v10(op, starts, ends, axes, steps, out):
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


class MoveTransposePastSlice_v1(RewriteRule, Verify):
    """Reorder transpose operations to follow slicing where applicable."""

    @staticmethod
    def pattern_v10(op, x, perm, starts, ends, axes, steps):
        return op.Slice(op.Transpose(x, perm=perm), starts, ends, axes, steps)

    @staticmethod
    def rewrite_v10(op, x, perm, starts, ends, axes, steps):
        return op.Transpose(
            op.Slice(
                x,
                starts,
                ends,
                op.Gather(
                    op.Constant(value_ints=perm.as_ints()),
                    axes
                ),
                steps
            ),
            perm=perm
        )


@tolerance
class MoveMatMulPastSlice_v1(RewriteRule, Verify):
    """Reorder matrix multiplications to follow slicing where applicable."""

    @staticmethod
    def pattern_v10(op, x, y, starts, ends, axes, steps):
        return op.Slice(op.MatMul(x, y), starts, ends, axes, steps)

    @staticmethod
    def rewrite_v10(op, x, y, starts, ends, axes, steps):
        # Turn all axis indices negative as it is easier to express the axis
        # selection for each input when counting from the back
        axes = op.Where(
            op.GreaterOrEqual(
                axes,
                op.Constant(value_int=0)
            ),
            op.Sub(
                axes,
                op.Size(
                    op.Shape(
                        op.MatMul(x, y)
                    )
                ),
            ),
            axes
        )

        # Map each output axis, counting from the back to a corresponding input
        # axis and the input side where it is coming from:
        #
        #   -1 -> -1 (rhs)
        #   -2 -> -2 (lhs)
        #   -n -> -n (lhs, rhs) for n >= 3, broadcastable batch dimensions
        #
        # Drop any axis which would be out of bounds without broadcasting the
        # inputs, i.e., when n > rank(<input>).
        lhs = op.And(
            op.Or(
                op.Equal(
                    axes,
                    op.Constant(value_int=-2)
                ),
                op.LessOrEqual(
                    axes,
                    op.Constant(value_int=-3)
                )
            ),
            op.LessOrEqual(
                op.Abs(axes),
                op.Size(op.Shape(x))  # ~ rank(x)
            )
        )

        rhs = op.And(
            op.Or(
                op.Equal(
                    axes,
                    op.Constant(value_int=-1)
                ),
                op.LessOrEqual(
                    axes,
                    op.Constant(value_int=-3)
                )
            ),
            op.LessOrEqual(
                op.Abs(axes),
                op.Size(op.Shape(y))  # ~ rank(y)
            )
        )

        # Replacement pattern: Slice inputs before the matrix multiplication,
        # selecting from the <starts,ends,axes,steps> according to rules above.
        return op.MatMul(
            op.Slice(
                x,
                # Note: Compress selects from an input given a condition
                op.Compress(starts, lhs),
                op.Compress(ends, lhs),
                op.Compress(axes, lhs),
                op.Compress(steps, lhs)
            ),
            op.Slice(
                y,
                op.Compress(starts, rhs),
                op.Compress(ends, rhs),
                op.Compress(axes, rhs),
                op.Compress(steps, rhs)
            )
        )
