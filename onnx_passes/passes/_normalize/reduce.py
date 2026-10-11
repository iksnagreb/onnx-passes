from onnx_passes.passes._base import RewriteRuleSetTemplate
from onnx_passes.passes._verify import Verify

import onnx_ir as ir
import numpy as np


class InferReduceAxes_v1(RewriteRuleSetTemplate, Verify):
    """Infer axes to reduce if axes are empty and noop_with_empty_axes=0."""

    patterns = (
        lambda op: op.ReduceL1,
        lambda op: op.ReduceL2,
        lambda op: op.ReduceLogSum,
        lambda op: op.ReduceLogSumExp,
        lambda op: op.ReduceMax,
        lambda op: op.ReduceMean,
        lambda op: op.ReduceMin,
        lambda op: op.ReduceProd,
        lambda op: op.ReduceSum,
        lambda op: op.ReduceSumSquare,
    )

    @staticmethod
    def pattern_v18(partial, op, x):
        return partial(op)(x, _allow_other_inputs=True, _outputs=["out"])

    @staticmethod
    def check_v18(context, x, out):
        # Collect original attributes (all Reduce* share supported and defaults
        # attributes) and infer the default noop_with_empty_axes.
        attributes = out.producer().attributes

        if not (noop_with_empty_axes := attributes.get("noop_with_empty_axes")):
            noop_with_empty_axes = ir.Attr(
                "noop_with_empty_axes", ir.AttributeType.INT, 0
            )

        # Handle both, missing optional and explicitly empty axes input if the
        # noop_with_empty_axes attribute is set to 0
        axes: ir.Value | None = None

        if len(out.producer().inputs) > 1:
            axes = out.producer().inputs[1]

        if axes is None or axes.shape is not None and np.prod(axes.shape) == 0:
            return noop_with_empty_axes.as_int() == 0

        return False

    @staticmethod
    def rewrite_v18(partial, op, x, out):
        return partial(op)(
            x,
            # Enumerate axes of the input x - this is always possible, even for
            # dynamically shaped inputs x
            op.Range(
                op.Constant(value_int=0),
                op.ReduceProd(
                    op.Reshape(
                        op.Size(
                            op.Shape(x)
                        ),
                        op.Constant(value_ints=[1])
                    ),
                    op.Constant(value_ints=[0])
                ),
                op.Constant(value_int=1),
            ),
            **out.producer().attributes
        )
