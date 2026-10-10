from onnx_passes.passes._base import RewriteRule
from onnx_passes.passes._verify import Verify

import onnx_ir as ir
import numpy as np


class InferConvDilations_v1(RewriteRule, Verify):
    """Infer convolution dilations if no attribute is given."""

    @staticmethod
    def pattern(op, x, w):
        return op.Conv(
            x, w, _allow_other_inputs=True, _outputs=["out"]
        )

    @staticmethod
    def check(context, x, w, out):
        if out.producer().attributes.get("dilations") is None:
            return w.shape is not None and w.shape.is_static()

        return False

    @staticmethod
    def rewrite(op, x, w, out):
        inputs = out.producer().inputs[2:]
        attributes = out.producer().attributes

        return op.Conv(
            x, w, *inputs, **attributes, dilations=len(w.shape[2:]) * [1]
        )


class InferConvKernelShape_v1(RewriteRule, Verify):
    """Infer convolution kernel size if no attribute is given."""

    @staticmethod
    def pattern(op, x, w):
        return op.Conv(
            x, w, _allow_other_inputs=True, _outputs=["out"]
        )

    @staticmethod
    def check(context, x, w, out):
        if out.producer().attributes.get("kernel_shape") is None:
            return w.shape is not None and w.shape.is_static()

        return False

    @staticmethod
    def rewrite(op, x, w, out):
        inputs = out.producer().inputs[2:]
        attributes = out.producer().attributes

        return op.Conv(
            x, w, *inputs, **attributes,
            kernel_shape=list(map(int, w.shape[2:]))
        )


class InferConvPads_v1(RewriteRule, Verify):
    """Infer convolution padding if no attribute is given."""

    @staticmethod
    def pattern(op, x, w, auto_pad):
        return op.Conv(
            x, w, auto_pad=auto_pad, _allow_other_inputs=True, _outputs=["out"]
        )

    @staticmethod
    def check(context, x, w, auto_pad, out):
        if auto_pad.as_string().startswith("SAME"):
            return w.shape is not None and w.shape.is_static()

        return False

    @staticmethod
    def rewrite(op, x, w, auto_pad, out):
        # Collect weights and bias inputs to the convolution operation and the
        # original attributes
        inputs = out.producer().inputs[2:]
        attributes = out.producer().attributes

        # Infer the default kernel shape and dilations (if not present) from the
        # weights parameter shape
        if (kernel_shape := attributes.get("kernel_shape")) is None:
            kernel_shape = ir.Attr(
                "kernel_shape", ir.AttributeType.INTS, w.shape[2:]
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

        # Pads per dimension such that the output has the same size as the input
        # and distribute pads to beginning/end with uneven amounts distributed
        # according to the SAME_* attribute.
        pads = []

        for s, d, k in zip(strides, dilations, kernel_shape):
            pads.append(d * (k - 1) + 1 - s)

        if auto_pad.as_string() == "SAME_LOWER":
            pads = [*[np.ceil(n / 2) for n in pads], *[n // 2 for n in pads]]

        if auto_pad.as_string() == "SAME_UPPER":
            pads = [*[n // 2 for n in pads], *[np.ceil(n / 2) for n in pads]]

        del attributes["auto_pad"]

        return op.Conv(x, w, *inputs, **attributes, pads=list(map(int, pads)))


class InferConvStrides_v1(RewriteRule, Verify):
    """Infer convolution strides if no attribute is given."""

    @staticmethod
    def pattern(op, x, w):
        return op.Conv(
            x, w, _allow_other_inputs=True, _outputs=["out"]
        )

    @staticmethod
    def check(context, x, w, out):
        if out.producer().attributes.get("strides") is None:
            return w.shape is not None and w.shape.is_static()

        return False

    @staticmethod
    def rewrite(op, x, w, out):
        inputs = out.producer().inputs[2:]
        attributes = out.producer().attributes

        return op.Conv(
            x, w, *inputs, **attributes, strides=len(w.shape[2:]) * [1]
        )


class RewritePaddedConvAsPadConv_v1(RewriteRule, Verify):
    """Rewrite padded convolution operation as explicit Pad-Conv combination."""

    @staticmethod
    def pattern(op, x, pads):
        return op.Conv(
            x, pads=pads, _allow_other_inputs=True, _outputs=["out"]
        )

    @staticmethod
    def check(context, x, pads, out):
        return any(pads.as_ints()) and x.shape is not None

    @staticmethod
    def rewrite(op, x, pads, out):
        # Collect weights and bias inputs to the convolution operation and the
        # original attributes, from which the pads are then deleted
        inputs = out.producer().inputs[1:]

        attributes = out.producer().attributes
        del attributes["pads"]

        spatial = len(pads := pads.as_ints()) // 2

        return op.Conv(
            # Explicit padding of the input, filling up pads for non-spatial
            # dimensions with zero
            op.Pad(
                x,
                op.Constant(
                    value_ints=[
                        *((len(x.shape) - spatial) * [0]), *pads[:spatial],
                        *((len(x.shape) - spatial) * [0]), *pads[spatial:],
                    ]
                ),
                op.CastLike(
                    op.Constant(value_float=0.0),
                    x
                )
            ),
            *inputs,
            **attributes
        )
