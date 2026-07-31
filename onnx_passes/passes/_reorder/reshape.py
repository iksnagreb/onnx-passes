from onnx_passes.passes._base import RewriteRule
from onnx_passes.passes._verify import Verify

from onnx_passes.traits.elementwise import produced_by_elementwise

import numpy as np
import onnx_ir as ir


class MoveElementwisePastReshape_v1(RewriteRule, Verify):
    """Reorder elementwise operations to follow reshape where applicable."""

    @staticmethod
    def pattern(op, shape):
        return op.Reshape(produced_by_elementwise, shape, _outputs=["out"])

    @staticmethod
    def rewrite(op, shape, out):
        # Find the elementwise operator which produces the input to the matched
        # reshape operator (the value level check guarantees this exists and is
        # indeed the node we are interested in).
        elementwise = out.producer().inputs[0].producer()

        # Collect the list of inputs to the elementwise operation with all
        # inputs expanded and reshaped to match the output shape.
        inputs = []

        for inp in elementwise.inputs:
            inputs.append(
                op.Reshape(
                    op.Expand(
                        inp,
                        op.Shape(elementwise.outputs[0])
                    ),
                    shape
                )
            )

        # Insert the replacement pattern with attributes transplanted from the
        # elementwise operator
        return op.op(elementwise.op_type, *inputs, **elementwise.attributes)


# Generate all permutations of an iterable (used to bruteforce possible shape
# permutations)
from itertools import permutations


def _reorder_transpose_reshape(perm, shape1, shape2):
    """Find a permutation-shape combination to reorder Transpose-Reshape."""

    # Enumerate the input tensor with indices and derive the expected
    # transpose-reshaped indices
    indices = np.arange(int(np.prod(shape1))).reshape(shape1)
    indices = indices.transpose(*perm).reshape(shape2)

    # Search through all possible permutations of the output shape to find one
    # that yields the same result when applied in reversed order
    for p in permutations(range(len(shape2))):
        shape = tuple(int(shape2[p.index(i)]) for i in range(len(p)))

        candidate = np.arange(int(np.prod(shape1))).reshape(shape1)
        candidate = candidate.reshape(shape).transpose(*p)

        try:
            if np.all(candidate == indices):
                return p, shape
        except ValueError:
            pass

    # Nothing found (could this actually ever happen?)
    return None


class MoveTransposePastReshape_v1(RewriteRule, Verify):
    """Reorder transpose operations to follow reshape where applicable."""

    @staticmethod
    def pattern(op, x, perm, shape):
        return op.Reshape(op.Transpose(x, perm=perm), shape)

    @staticmethod
    def check(op, x, perm, shape):
        if (shape1 := x.shape) is None or not shape1.is_static():
            return False

        if (shape2 := ir.convenience.get_const_tensor(shape)) is None:
            return False

        shape2, perm = shape2.numpy(), perm.as_ints()

        return _reorder_transpose_reshape(perm, shape1, shape2) is not None

    @staticmethod
    def rewrite(op, x, perm, shape):
        # Extract static shapes and permutation from the match context and find
        # the valid reordering shape and permutation
        shape1 = x.shape
        shape2 = ir.convenience.get_const_tensor(shape).numpy()  # noqa: const
        perm = perm.as_ints()

        perm, shape = _reorder_transpose_reshape(perm, shape1, shape2)  # noqa

        return op.Transpose(
            op.Reshape(
                x,
                op.Constant(value_ints=shape)
            ),
            perm=perm,
        )
