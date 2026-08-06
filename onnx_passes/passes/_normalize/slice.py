from onnx_passes.passes._base import RewriteRule
from onnx_passes.passes._verify import Verify

import numpy as np


class InferSliceAxesAndSteps_v1(RewriteRule, Verify):
    """Infers axes to slice if no axes input or attribute is given."""

    @staticmethod
    def pattern_v10(op, x):
        return op.Slice(x, _allow_other_inputs=True, _outputs=["y"])

    @staticmethod
    def check_v10(op, x, y):
        # Missing axes input (either both optionals are missing or the optional
        # axes input is explicitly set to None)
        if len(inputs := y.producer().inputs) < 4 or inputs[3] is None:
            return x.shape is not None and x.shape.is_static()

        # Missing steps input (either the input is missing from the list or the
        # optional steps input is explicitly set to None)
        if len(inputs := y.producer().inputs) < 5 or inputs[4] is None:
            return x.shape is not None and x.shape.is_static()

        # Everything seems to be present - do not rewrite
        return False

    @staticmethod
    def rewrite_v10(op, x, y):
        # Collect the existing inputs and extend by Nones to fill up to the full
        # list including optionals
        inputs = [
            *y.producer().inputs, *((5 - len(y.producer().inputs)) * [None])
        ]

        # Explicitly fill in the missing optional axes (enumerating all axes)
        # and steps (all ones) inputs if missing
        if inputs[3] is None:
            inputs[3] = op.Constant(value_ints=np.arange(len(x.shape)))

        if inputs[4] is None:
            inputs[4] = op.Expand(op.Constant(value_int=1), op.Shape(inputs[3]))

        return op.Slice(*inputs)
