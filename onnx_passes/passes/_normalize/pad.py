from onnx_passes.passes._base import RewriteRule
from onnx_passes.passes._verify import Verify

import onnx_ir as ir
import numpy as np


class InferPadAxes_v1(RewriteRule, Verify):
    """Infer axes to pad if no axes input is given."""

    @staticmethod
    def pattern_v18(op, x, pads, mode):
        return op.Pad(
            x, pads, mode=mode, _allow_other_inputs=True, _outputs=["out"]
        )

    @staticmethod
    def check_v18(op, x, pads, mode, out):
        return len(inputs := out.producer().inputs) < 4 or inputs[3] is None

    @staticmethod
    def rewrite_v18(op, x, pads, mode, out):
        constant_value = None

        if len(inputs := out.producer().inputs) == 3:
            constant_value = inputs[2]

        return op.Pad(
            x,
            pads,
            constant_value,
            # Enumerate all axes of the input rank
            op.Range(
                op.Constant(value_int=0),
                op.Size(op.Shape(x)),
                op.Constant(value_int=1)
            ),
            mode=mode
        )


class SortPadAxes_v1(RewriteRule, Verify):
    """Sort padding axes and pads in increasing axis order."""

    @staticmethod
    def pattern_v18(op, x, pads, const_value, axes, mode):
        return op.Pad(x, pads, const_value, axes, mode=mode)

    @staticmethod
    def check_v18(context, x, pads, const_value, axes, mode):
        if (axes := ir.convenience.get_const_tensor(axes)) is not None:
            return np.any(sorted(axes.numpy()) != axes.numpy())

        return False

    @staticmethod
    def rewrite_v18(op, x, pads, const_value, axes, mode):
        # Sort axes in increasing order and also keep the indices to sort each
        # half of the pads (left/right pads) into the same order
        axes, indices = op.TopK(axes, op.Shape(axes), largest=0, _outputs=2)
        starts, ends = op.Split(pads, num_outputs=2, _outputs=2)

        return op.Pad(
            x,
            op.Concat(
                op.Gather(
                    starts,
                    indices
                ),
                op.Gather(
                    ends,
                    indices
                ),
                axis=0
            ),
            const_value,
            axes,
            mode=mode
        )


class InferPadConstValue_v1(RewriteRule, Verify):
    """Infer constant value to pad if no const_value input is given."""

    @staticmethod
    def pattern(op, x, pads, mode):
        return op.Pad(
            x, pads, mode=mode, _allow_other_inputs=True, _outputs=["out"]
        )

    @staticmethod
    def check(op, x, pads, mode, out):
        if mode.as_string() == "constant":
            return len(inputs := out.producer().inputs) < 3 or inputs[2] is None
        return False

    @staticmethod
    def rewrite(op, x, pads, mode, out):
        return op.Pad(
            x,
            pads,
            op.CastLike(
                op.Constant(value_int=0),
                x
            ),
            mode=mode
        )

    @staticmethod
    def rewrite_v18(op, x, pads, mode, out):
        maybe_axes = []

        if len(inputs := out.producer().inputs) == 4:
            maybe_axes = [inputs[3]]

        return op.Pad(
            x,
            pads,
            op.CastLike(
                op.Constant(value_int=0),
                x
            ),
            *maybe_axes,
            mode=mode
        )
