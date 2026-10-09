from onnx_passes.passes._base import RewriteRule
from onnx_passes.passes._verify import Verify, tolerance

from onnx_passes.ops import DOMAIN as CUSTOM_DOMAIN

import onnx_ir as ir
import numpy as np


@tolerance
class LowerConvToMatMul_v1(RewriteRule, Verify):
    """Lower convolution operations (without padding) as Im2Col-MatMul.

    Convolutions are adapted from channels-first, i.e., N x C x D1 x D2 x... to
    channels-last, i.e., N x D1 x D2 x ... x Dn x C layout via a transpose pair.

    Convolution bias is extracted as a standalone operator and group convolution
    is implemented via Split and Concat operators.
    """

    @staticmethod
    def pattern(op, x, w):
        return op.Conv(x, w, _allow_other_inputs=True, _outputs=["out"])

    @staticmethod
    def check(context, x, w, out):
        attributes = out.producer().attributes

        # Convolution lowering does not handle padded convolutions - should be
        # normalized to Pad-Conv pattern first
        if (auto_pad := attributes.get("auto_pad")) is not None:
            if auto_pad.as_string() not in {"NOTSET", "VALID"}:
                return False

        if (pads := attributes.get("pads")) is not None:
            if np.any(pads.as_ints() != 0):
                return False

        # Convolution lowering does not handle grouped convolutions - should be
        # lowered to Split-Conv pattern first
        if (group := attributes.get("group")) is not None:
            if group.as_int() != 1:
                return False

        # Weight shape must be static to infer the kernel shape if not given
        # explicitly as a node attribute.
        return w.shape is not None and w.shape.is_static()

    @staticmethod
    def rewrite(op, x, w, out):
        # Convert the input from channels-first as used by the Conv operator to
        # channels-last layout as used by Im2Col and MatMul
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

        # Collect weights and bias inputs to the convolution operation and the
        # original attributes
        bias = (inputs[2]) if len(inputs := out.producer().inputs) > 2 else None
        attributes = out.producer().attributes

        # Infer the default kernel shape and dilations (if not present) from the
        # weights parameter shape
        if (kernel_shape := attributes.get("kernel_shape")) is None:
            attributes["kernel_shape"] = kernel_shape = ir.Attr(
                "kernel_shape", ir.AttributeType.INTS, w.shape[2:]
            )

        kernel_shape = kernel_shape.as_ints()

        if "dilations" not in attributes:
            attributes["dilations"] = ir.Attr(
                "dilations", ir.AttributeType.INTS, len(kernel_shape) * [1]
            )

        if "strides" not in attributes:
            attributes["strides"] = ir.Attr(
                "strides", ir.AttributeType.INTS, len(kernel_shape) * [1]
            )

        # Delete padding and grouping-related attributes from the Conv operator
        # which are not handled by the lowered Im2Col
        for key in {"auto_pad", "pads", "group"}:
            try:
                del attributes[key]
            except KeyError:
                pass

        # Lowered convolution: Inputs generated via Im2Col and MatMul with
        # flattened filter kernel weights
        y = op.MatMul(
            # Generate sliding windows from the input using the custom Im2Col
            # operator
            op.Im2Col(
                x, **attributes, _domain=CUSTOM_DOMAIN, _version=2
            ),
            # Shuffle and flatten weights into channels-last (input channel in
            # the innermost dimension) layout
            op.Transpose(
                op.Reshape(
                    op.Transpose(
                        op.Reshape(
                            w,
                            op.Concat(
                                o := op.Shape(w, start=0, end=1),
                                c := op.Shape(w, start=1, end=2),
                                op.ReduceProd(
                                    op.Shape(w, start=2)
                                ),
                                axis=0
                            )
                        ),
                        perm=[0, 2, 1]
                    ),
                    op.Concat(
                        o,
                        op.Mul(
                            c,
                            op.ReduceProd(
                                op.Shape(w, start=2)
                            ),
                        ),
                        axis=0
                    )
                )
            )
        )

        # Insert standalone addition for optional convolution bias along the
        # channel dimension
        if bias is not None:
            y = op.Add(y, bias)

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
