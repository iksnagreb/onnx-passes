from onnx_passes.passes._base import RewriteRule, Transformation, Sequential
from onnx_passes.passes._verify import Verify

from onnx_passes.traits.elementwise import produced_by_elementwise
from onnx_passes.traits.expand import produced_by_expand

import onnx_ir as ir
import numpy as np


class MoveExpandPastElementwise_v1(RewriteRule, Verify):
    """Reorder Expand operations to follow elementwise operations."""

    @staticmethod
    def pattern(op):
        return op.submodule("")(_allow_other_inputs=True, _outputs=["out"])

    @staticmethod
    def check(context, out):
        # Accept rewrite as soon as any input is expanded and the overall
        # output shape is static and this is not all constant
        if not produced_by_elementwise(context, out):
            return False

        for x in out.producer().inputs:
            if produced_by_expand(context, x):
                if any(ir.convenience.get_const_tensor(v) is None
                       for v in x.producer().inputs):
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
                if any(ir.convenience.get_const_tensor(v) is None
                       for v in inp.producer().inputs):
                    inp = inp.producer().inputs[0]

            inputs.append(inp)

        # Insert the replacement pattern with attributes transplanted from the
        # elementwise operator and final output expanded
        return op.Expand(
            op.op(
                elementwise.op_type, *inputs, **elementwise.attributes
            ),
            op.Constant(
                value_ints=ir.Attr(
                    "value_ints", ir.AttributeType.INTS, out.shape[:]
                )
            )
        )


def _is_unsqueeze_of(shape, other):
    """Test whether shape can be derived from other by unsqueezing."""
    return [size for size in shape if size != 1] == list(other)


def _reorder_expand_reshape(shape, expand, reshape):
    """Find shapes which reorder Expand-Reshape if possible."""

    if _is_unsqueeze_of(reshape, expand):
        # Find the axes that need to be unsqueezed from the original and
        # expanded shape
        unsqueeze = list(np.where(np.asarray(reshape) == 1)[0])
        unsqueeze = np.expand_dims(np.empty(shape), unsqueeze).shape

        return tuple(map(int, reshape)), tuple(map(int, unsqueeze))

    return None


class MoveExpandPastReshape_v1(RewriteRule, Verify):
    """Reorder Expand operations to follow Reshape where applicable."""

    @staticmethod
    def pattern(op, x, expand, reshape):
        return op.Reshape(op.Expand(x, expand), reshape)

    @staticmethod
    def check(context, x, expand, reshape):
        if (expand := ir.convenience.get_const_tensor(expand)) is None:
            return False

        if (reshape := ir.convenience.get_const_tensor(reshape)) is None:
            return False

        if (shape := x.shape) is None or shape.is_dynamic():
            return False

        expand = expand.numpy()
        reshape = reshape.numpy()

        return _reorder_expand_reshape(shape, expand, reshape) is not None

    @staticmethod
    def rewrite(op, x, expand, reshape):
        # Extract static shapes from the match context and find the valid
        # reordering shape and expansion
        expand = ir.convenience.get_const_tensor(expand).numpy()
        reshape = ir.convenience.get_const_tensor(reshape).numpy()

        expand, reshape = _reorder_expand_reshape(  # noqa: not None
            x.shape, expand, reshape
        )

        return op.Expand(
            op.Reshape(
                x, op.Constant(
                    value_ints=ir.Attr(
                        "value_ints", ir.AttributeType.INTS, reshape
                    )
                )
            ),
            op.Constant(
                value_ints=ir.Attr(
                    "value_ints", ir.AttributeType.INTS, expand
                )
            )
        )


class ReorderExpandLoop_v1(Sequential, Transformation):
    """Exhaustively apply expand reordering transformations."""

    passes = [
        MoveExpandPastElementwise_v1,
        MoveExpandPastReshape_v1,
    ]

    exhaustive = True
