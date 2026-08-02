from onnx_passes.passes._base import RewriteRule
from onnx_passes.passes._verify import Verify, tolerance

from onnx_passes.traits.elementwise import produced_by_elementwise
from onnx_passes.traits.reduction import produced_by_reduction

import numpy as np
import onnx_ir as ir


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
        return op.op(elementwise.op_type, *inputs, **elementwise.attributes)


@tolerance
class MoveMatMulPastTranspose_v1(RewriteRule, Verify):
    """Reorder matrix multiplication to follow transpose where applicable."""

    @staticmethod
    def pattern(op, x, y, perm):
        return op.Transpose(op.MatMul(x, y), perm=perm)

    @staticmethod
    def check(op, x, y, perm):
        if x.shape is not None and x.shape.is_static():
            if y.shape is not None and y.shape.is_static():
                perm = np.array(perm.as_ints()) - np.int64(len(perm.as_ints()))
                return tuple(perm[-2:]) in {(-1, -2), (-2, -1)}
        return False

    @staticmethod
    def rewrite(op, x, y, perm):
        # Permutation as numpy array with reverse indexing (simplifies checks
        # for swapping the final two axes)
        perm = np.array(perm.as_ints()) - np.int64(len(perm.as_ints()))

        # The replacement pattern is decided by the shape signature of the
        # inputs roughly following the broadcasting behavior of the matmul
        signature = (len(x.shape), len(y.shape))

        # 2d - 2d: Transpose both sides using the same 2d permutation
        if signature == (2, 2):
            x = op.Transpose(x, perm=[int(i) + len(perm) for i in perm])
            y = op.Transpose(y, perm=[int(i) + len(perm) for i in perm])

        # 1d - Nd: Transpose right hand side with added permutation axis as this
        # product yields (N-1)d at outputs.
        if signature[0] == 1 and signature[1] > 2:
            x = x

            # Swap the final two axes (one more axis than the output) first, if
            # the output permutes those axes
            if tuple(perm[-2:]) == (-1, -2):
                y = op.Transpose(
                    y, perm=[*range(len(perm) - 1), len(perm), len(perm) - 1]
                )

            y = op.Transpose(
                y, perm=[*[int(i) + len(perm) for i in perm], len(perm)]
            )

        # Nd - 1d: Transpose left hand side with added permutation axis as this
        # product yields (N-1)d at outputs.
        if signature[0] > 2 and signature[1] == 1:
            x = op.Transpose(
                x, perm=[*[int(i) + len(perm) for i in perm], len(perm)]
            )

            # Swap the final two axes (one more axis than the output) last, if
            # the output permutes those axes
            if tuple(perm[-2:]) == (-1, -2):
                x = op.Transpose(
                    x, perm=[*range(len(perm) - 1), len(perm), len(perm) - 1]
                )

            y = y

        # 2d - Nd: Transpose the right hand side where the output permutation
        # directly applies. If the last two axes are swapped, also permute the
        # left hand side.
        if signature[0] == 2 and signature[1] > 2:
            if tuple(perm[-2:]) == (-1, -2):
                x = op.Transpose(x)

            y = op.Transpose(y, perm=[int(i) + len(perm) for i in perm])

        # Nd - 2d: Transpose the left hand side where the output permutation
        # directly applies. If the last two axes are swapped, also permute the
        # right hand side.
        if signature[0] > 2 and signature[1] == 2:
            if tuple(perm[-2:]) == (-1, -2):
                y = op.Transpose(y)

            x = op.Transpose(x, perm=[int(i) + len(perm) for i in perm])

        # Nd - Md: Transpose both inputs after unsqueezing the shorter to match
        # the rank of the longer. Squeeze leading dimensions after transpose.
        if signature[0] > 2 and signature[1] > 2:
            unsqueezed_x = ...
            unsqueezed_y = ...

            if signature[0] < signature[1]:
                x = op.Unsqueeze(
                    x,
                    unsqueezed_x := op.Constant(
                        value_ints=list(range(signature[1] - signature[0]))
                    )
                )

            if signature[1] < signature[0]:
                y = op.Unsqueeze(
                    y,
                    unsqueezed_y := op.Constant(
                        value_ints=list(range(signature[0] - signature[1]))
                    )
                )

            x = op.Transpose(x, perm=[int(i) + len(perm) for i in perm])
            y = op.Transpose(y, perm=[int(i) + len(perm) for i in perm])

            if signature[0] < signature[1]:
                x = op.Squeeze(  # noqa: Duplicate, see below...
                    x,
                    op.Range(
                        op.Constant(value_int=0),
                        op.Min(
                            op.ArgMin(
                                op.Cast(
                                    op.Equal(
                                        op.Shape(x),
                                        op.Constant(value_int=1)
                                    ),
                                    to=ir.DataType.UINT8
                                )
                            ),
                            op.Size(unsqueezed_x)
                        ),
                        op.Constant(value_int=1),
                    )
                )
            if signature[1] < signature[0]:
                y = op.Squeeze(  # noqa: Duplicate, see below...
                    y,
                    op.Range(
                        op.Constant(value_int=0),
                        op.Min(
                            op.ArgMin(
                                op.Cast(
                                    op.Equal(
                                        op.Shape(y),
                                        op.Constant(value_int=1)
                                    ),
                                    to=ir.DataType.UINT8
                                )
                            ),
                            op.Size(unsqueezed_y)
                        ),
                        op.Constant(value_int=1),
                    )
                )

        # Swap the order of inputs if permuting the final two axes as
        #   (x @ y)^T = y^T @ x^T for 2d matrices
        if tuple(perm[-2:]) == (-1, -2):
            return op.MatMul(y, x)

        return op.MatMul(x, y)


class MoveReducePastTranspose_v1(RewriteRule, Verify):
    """Reorder reduction operations to follow transpose where applicable."""

    @staticmethod
    def pattern(op, perm):
        return op.Transpose(produced_by_reduction, perm=perm, _outputs=["out"])

    @staticmethod
    def check(op, perm, out):
        # Find the reduction operator which produces the input to the matched
        # transpose operator (the value level check guarantees this exists and
        # is indeed the node we are interested in).
        reduce = out.producer().inputs[0].producer()

        # If reduced dimensions are removed from the shape, the permutation will
        # not match the input rank. To fill in missing dimensions in this case,
        # the input shape and reduction axes must be statically.
        x, axes = reduce.inputs

        if (keepdims := reduce.attributes.get("keepdims", None)) is None:
            keepdims = ir.Attr("keepdims", ir.AttributeType.INT, 1)

        if keepdims.as_int() == 0:
            if x.shape is not None and x.shape.is_static():
                return ir.convenience.get_const_tensor(axes) is not None

        return True

    @staticmethod
    def rewrite(op, perm, out):
        # Find the reduction operator which produces the input to the matched
        # transpose operator (the value level check guarantees this exists and
        # is indeed the node we are interested in).
        reduce = out.producer().inputs[0].producer()

        if (keepdims := reduce.attributes.get("keepdims", None)) is None:
            keepdims = ir.Attr("keepdims", ir.AttributeType.INT, 1)

        perm = perm.as_ints()
        x, axes = reduce.inputs

        # If Reduce deletes the reduction axes, the transpose permutation must
        # be adjusted by filling in missing dimensions at the end.
        if keepdims.as_int() == 0:
            # Get the static reduction axes and normalize to positive indices as
            # permutations cannot use negative indexing
            axes = ir.convenience.get_const_tensor(axes).numpy()  # noqa: Const
            axes = np.where(axes < 0, len(x.shape) + axes, axes)

            # Insert reduction axes into the permutation and adjust all axes
            # already present by shifting them up accounting for dimensions
            # inserted below
            for axis in axes:
                perm = np.asarray(perm)
                perm = list(np.where(axis <= perm, perm + 1, perm))
                perm.insert(axis, axis)

            perm = [int(i) for i in perm]

            # When the output is a full reduction to a scalar, the permutation
            # might be empty (if empty axes and noop_with_empty_axes)
            if not perm:
                perm = list(range(len(x.shape)))

        # Insert the replacement pattern with attributes transplanted from the
        # reduction operator and input and reduction axes permuted
        return op.op(
            reduce.op_type,
            op.Transpose(
                reduce.inputs[0],
                perm=perm
            ),
            # Transpose axes via gathering from the permutation list
            op.Gather(
                op.Constant(value_ints=perm),
                reduce.inputs[1]
            ),
            **reduce.attributes
        )
