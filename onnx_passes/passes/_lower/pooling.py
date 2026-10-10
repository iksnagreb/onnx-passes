from onnx_passes.passes._base import RewriteRuleSetTemplate
from onnx_passes.passes._verify import Verify, tolerance

from onnx_passes.ops import DOMAIN as CUSTOM_DOMAIN

import onnx_ir as ir
import numpy as np


def reduce_lp(op):
    """Generate ONNX script Lp-norm reduction."""

    def script(x, axes, p: ir.Attr, keepdims: ir.Attr):
        return op.Pow(
            op.ReduceSum(
                op.Pow(
                    op.Abs(x),
                    op.Constant(value_float=float(p.as_int()))
                ),
                axes,
                keepdims=keepdims
            ),
            op.Constant(value_float=1.0 / p.as_int())
        )

    return script


@tolerance
class LowerPoolToReduce_v1(RewriteRuleSetTemplate, Verify):
    """Lower pooling operations (without padding) as Im2Col-Reduce*.

    Pooling is adapted from channels-first, i.e., N x C x D1 x D2 x... to
    channels-last, i.e., N x D1 x D2 x ... x Dn x C layout via a transpose pair.
    """

    patterns = (
        lambda op: (op.MaxPool, op.ReduceMax),
        lambda op: (op.AveragePool, op.ReduceMean),
        lambda op: (op.LpPool, reduce_lp(op))
    )

    @staticmethod
    def pattern(partial, op, x):
        return partial(op)[0](x, _allow_other_inputs=True, _outputs=["out"])

    @staticmethod
    def check(context, x, out):
        attributes = out.producer().attributes

        # Pooling lowering does not handle padded pooling - should be normalized
        # to Pad-*Pool pattern first
        if (auto_pad := attributes.get("auto_pad")) is not None:
            if auto_pad.as_string() not in {"NOTSET", "VALID"}:
                return False

        if (pads := attributes.get("pads")) is not None:
            if np.any(pads.as_ints() != 0):
                return False

        return True

    @staticmethod
    def rewrite(partial, op, x, out):
        # Convert the input from channels-first as used by the *Pool operator to
        # channels-last layout as used by Im2Col and Reduce*
        x = op.Reshape(
            op.Transpose(
                op.Reshape(
                    x,
                    op.Concat(
                        op.Shape(x, start=0, end=1),
                        op.Shape(x, start=1, end=2),
                        op.ReduceProd(
                            op.Shape(x, start=2)
                        ),
                        axis=0
                    )
                ),
                perm=[0, 2, 1]
            ),
            op.Concat(
                op.Shape(x, start=0, end=1),  # N
                op.Shape(x, start=2),  # D1 x D2 x ... Dn
                op.Shape(x, start=1, end=2),  # C
                axis=0
            )
        )

        attributes = out.producer().attributes

        # Infer the default kernel shape and dilations (if not present) from the
        # weights parameter shape
        if (kernel_shape := attributes.get("kernel_shape")) is None:
            kernel_shape = ir.Attr(
                "kernel_shape", ir.AttributeType.INTS, x.shape[2:]
            )

        kernel_shape = kernel_shape.as_ints()

        if (dilations := attributes.get("dilations")) is None:
            dilations = ir.Attr(
                "dilations", ir.AttributeType.INTS, len(kernel_shape) * [1]
            )

        dilations = dilations.as_ints()

        if (strides := attributes.get("strides")) is None:
            strides = ir.Attr(
                "strides", ir.AttributeType.INTS, len(kernel_shape) * [1]
            )

        strides = strides.as_ints()

        # Delete padding-related attributes from the Pool operator which are not
        # handled by the lowered Im2Col
        for key in {"auto_pad", "pads", "ceil_mode"}:
            try:
                del attributes[key]
            except KeyError:
                pass

        # Delete specific pooling operator attributes which are not handled by
        # the lowered Im2Col
        for key in {"count_include_pad", "storage_order"}:
            try:
                del attributes[key]
            except KeyError:
                pass

        im2col_attributes = {
            "kernel_shape": kernel_shape,
            "strides": strides,
            "dilations": dilations
        }

        # Delete Im2Col operator attributes which are not handled by the lowered
        # Reduce* operation
        for key in {"kernel_shape", "strides", "dilations"}:
            try:
                del attributes[key]
            except KeyError:
                pass

        # Lowered pooling: Inputs generated via Im2Col reduction over the window
        # via the corresponding Reduce* operator
        y = partial(op)[1](
            # Generate sliding windows from the input using the custom Im2Col
            # operator and disentangle the channel and kernel as pooling does
            # not reduce along the channel axis.
            op.Reshape(
                windows := op.Im2Col(
                    x, **im2col_attributes, _domain=CUSTOM_DOMAIN, _version=2
                ),
                op.Concat(
                    op.Shape(windows, end=-1),
                    op.ReduceProd(
                        op.Constant(value_ints=kernel_shape)
                    ),
                    op.Shape(x, start=-1),
                    axis=0
                )
            ),
            # Reduce over the flattened kernel dimensions (now second to last
            # axis) and get rid of this axis.
            op.Constant(value_ints=[-2]),
            keepdims=0,
            # Forward all remaining attributes of the *Pool operator node to the
            # lowered Reduce* operator, e.g., the LpPool p attribute.
            **attributes
        )

        # Convert back to channels-first layout to preserve the overall shape of
        # inputs and outputs in the graph
        return op.Reshape(
            op.Transpose(
                op.Reshape(
                    y,
                    op.Concat(
                        op.Shape(y, start=0, end=1),
                        op.ReduceProd(
                            op.Shape(y, start=1, end=-1)
                        ),
                        op.Shape(y, start=-1),
                        axis=0
                    )
                ),
                perm=[0, 2, 1]
            ),
            op.Concat(
                op.Shape(y, start=0, end=1),  # N
                op.Shape(y, start=-1),  # C'
                op.Shape(y, start=1, end=-1),  # D1' x D2' x ... Dn'
                axis=0
            )
        )
