from onnx_passes.passes._base import RewriteRule
from onnx_passes.passes._verify import Verify, tolerance

from onnx_passes.traits.elementwise import produced_by_elementwise
from onnx_passes.traits.reduction import produced_by_reduction

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
                op.Max(
                    op.Size(
                        op.Shape(
                            op.MatMul(x, y)
                        )
                    ),
                    op.Size(
                        op.Shape(x)
                    ),
                    op.Size(
                        op.Shape(y)
                    )
                )
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

        # Remove any broadcastable axes with singleton 1 in the shape as slicing
        # these from any non-zero start results in out of bounds access.
        singleton_lhs = op.Equal(
            op.Gather(
                op.Shape(x),
                op.Compress(
                    axes,
                    lhs
                )
            ),
            op.Constant(value_int=1)
        )

        singleton_rhs = op.Equal(
            op.Gather(
                op.Shape(y),
                op.Compress(
                    axes,
                    rhs
                )
            ),
            op.Constant(value_int=1)
        )

        # Replacement pattern: Slice inputs before the matrix multiplication,
        # selecting from the <starts,ends,axes,steps> according to rules above.
        return op.MatMul(
            op.Slice(  # noqa: Dumplicate
                x,
                # Note: Compress selects from an input given a condition
                op.Compress(op.Compress(starts, lhs), op.Not(singleton_lhs)),
                op.Compress(op.Compress(ends, lhs), op.Not(singleton_lhs)),
                op.Compress(op.Compress(axes, lhs), op.Not(singleton_lhs)),
                op.Compress(op.Compress(steps, lhs), op.Not(singleton_lhs)),
            ),
            op.Slice(  # noqa: Dumplicate
                y,
                op.Compress(op.Compress(starts, rhs), op.Not(singleton_rhs)),
                op.Compress(op.Compress(ends, rhs), op.Not(singleton_rhs)),
                op.Compress(op.Compress(axes, rhs), op.Not(singleton_rhs)),
                op.Compress(op.Compress(steps, rhs), op.Not(singleton_rhs)),
            )
        )


class MoveReducePastSlice_v1(RewriteRule, Verify):
    """Reorder reduction operations to follow slicing where applicable."""

    @staticmethod
    def pattern(op, starts, ends, axes, steps):
        return op.Slice(
            produced_by_reduction, starts, ends, axes, steps, _outputs=["out"]
        )

    @staticmethod
    def rewrite(op, starts, ends, axes, steps, out):
        # Find the reduction operator which produces the input to the matched
        # slice operator (the value level check guarantees this exists and
        # is indeed the node we are interested in).
        reduce = out.producer().inputs[0].producer()

        if (keepdims := reduce.attributes.get("keepdims", None)) is None:
            keepdims = ir.Attr("keepdims", ir.AttributeType.INT, 1)

        x, reduction_axes = reduce.inputs
        reduced = reduce.outputs[0]

        new_axes = axes

        # If reduce deletes the reduction axes, the slice axes must be adjusted
        # by reindexing to account for missing dimensions.
        if keepdims.as_int() == 0:
            new_axes = op.Gather(
                # Select non-reduction axes from the input shape
                op.Compress(
                    # Generate all input axes
                    op.Range(
                        op.Constant(value_int=0),
                        op.Size(op.Shape(x)),
                        op.Constant(value_int=1)
                    ),
                    # Mark non-reduction axes
                    op.Not(
                        op.Cast(
                            op.ReduceMax(
                                op.OneHot(
                                    reduction_axes,
                                    op.Size(op.Shape(x)),
                                    op.Constant(value_ints=[0, 1])
                                ),
                                op.Constant(value_ints=[0]),
                                keepdims=0
                            ),
                            to=ir.DataType.BOOL
                        )
                    )
                ),
                axes
            )

        # Avoid slicing reduced axes - these values must contribute to the
        # reduction. Extend slice for reduced axes to the full extent.
        new_starts = op.Where(
            # Reduced axes are those for which the size before/after reduction
            # differs. With keepdims=0 this will always be false as it is
            # impossible to express a slice over reduced axes.
            reduced_axes := op.Not(
                op.Equal(
                    op.Gather(
                        op.Shape(x),
                        new_axes
                    ),
                    op.Gather(
                        op.Shape(reduced),
                        axes
                    )
                )
            ),
            op.Constant(value_int=0),
            starts
        )

        new_ends = op.Where(
            reduced_axes,
            op.Gather(
                op.Shape(x),
                new_axes
            ),
            ends
        )

        new_steps = op.Where(
            reduced_axes,
            op.Constant(value_int=1),
            steps
        )

        # Insert the replacement pattern with attributes transplanted from the
        # reduction operator and input and reduction axes permuted
        return op.Where(
            # Result of slicing might be an empty tensor: In this case it is not
            # safe to reorder due to shape mismatch but also straightforward to
            # replace the pattern with an empty constant tensor.
            op.Equal(
                op.Size(
                    empty := op.Slice(
                        op.ConstantOfShape(op.Shape(reduced)),
                        starts,
                        ends,
                        axes,  # Note: Use the old axes/starts/ends/steps
                        steps
                    )
                ),
                op.Constant(value_int=0)
            ),
            empty,
            # Insert the reordered Slice-Reduce replacement pattern
            op.op(
                reduce.op_type,
                op.Slice(
                    x,
                    new_starts,
                    new_ends,
                    new_axes,
                    new_steps
                ),
                reduction_axes,
                **reduce.attributes
            )
        )
