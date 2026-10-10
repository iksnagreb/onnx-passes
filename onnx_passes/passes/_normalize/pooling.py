from onnx_passes.passes._base import RewriteRule, RewriteRuleSetTemplate
from onnx_passes.passes._verify import Verify, tolerance

import onnx_ir as ir
import numpy as np


class ExplicitGlobalPool_v1(RewriteRuleSetTemplate, Verify):
    """Rewrite Global* pooling as regular pooling spanning the whole input size.
    """

    patterns = (
        lambda op: (op.GlobalMaxPool, op.MaxPool),
        lambda op: (op.GlobalAveragePool, op.AveragePool),
        lambda op: (op.GlobalLpPool, op.LpPool),
    )

    @staticmethod
    def pattern(partial, op, x):
        return partial(op)[0](x, _outputs=["out"])

    @staticmethod
    def check(context, x, out):
        return x.shape is not None and x.shape.is_static()

    @staticmethod
    def rewrite(partial, op, x, out):
        return partial(op)[1](
            x, kernel_shape=x.shape[2:], **out.producer().attributes
        )


class InferPoolingDilations_v1(RewriteRuleSetTemplate, Verify):
    """Infer pooling dilations if no attribute is given."""

    patterns = (
        lambda op: op.MaxPool,
        lambda op: op.AveragePool,
        lambda op: op.LpPool,
    )

    @staticmethod
    def pattern(partial, op, x, kernel_shape):
        return partial(op)(x, kernel_shape=kernel_shape, _outputs=["out"])

    @staticmethod
    def check(context, x, kernel_shape, out):
        return out.producer().attributes.get("dilations") is None

    @staticmethod
    def rewrite(partial, op, x, kernel_shape, out):
        attributes = out.producer().attributes

        return partial(op)(
            x, **attributes, dilations=len(kernel_shape.as_ints()) * [1]
        )


class InferPoolingPads_v1(RewriteRuleSetTemplate, Verify):
    """Infer pooling padding if no attribute is given."""

    patterns = (
        # Note: ONNX reference seems to be wrong for SAME_* padding MaxPool:
        #   https://github.com/onnx/onnx/pull/8553
        # lambda op: op.MaxPool,
        lambda op: op.AveragePool,
        lambda op: op.LpPool,
    )

    @staticmethod
    def pattern(partial, op, x, kernel_shape, auto_pad):
        return partial(op)(
            x, kernel_shape=kernel_shape, auto_pad=auto_pad, _outputs=["out"]
        )

    @staticmethod
    def check(context, x, kernel_shape, auto_pad, out):
        if auto_pad.as_string().startswith("SAME"):
            if x.shape is not None and x.shape.is_static():
                return out.shape is not None and out.shape.is_static()

        return False

    @staticmethod
    def rewrite(partial, op, x, kernel_shape, auto_pad, out):
        attributes = out.producer().attributes

        # Kernel shape and dilations and strides inferred (if not present) from
        # the kernel shape
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

        xs = x.shape[2:]
        os = out.shape[2:]

        for o, i, s, d, k in zip(os, xs, strides, dilations, kernel_shape):
            pads.append((o - 1) * s + (d * (k - 1) + 1) - i)

        if auto_pad.as_string() == "SAME_LOWER":
            pads = [*[np.ceil(n / 2) for n in pads], *[n // 2 for n in pads]]

        if auto_pad.as_string() == "SAME_UPPER":
            pads = [*[n // 2 for n in pads], *[np.ceil(n / 2) for n in pads]]

        del attributes["auto_pad"]

        return partial(op)(x, **attributes, pads=list(map(int, pads)))


class InferPoolingStrides_v1(RewriteRuleSetTemplate, Verify):
    """Infer pooling strides if no attribute is given."""

    patterns = (
        lambda op: op.MaxPool,
        lambda op: op.AveragePool,
        lambda op: op.LpPool,
    )

    @staticmethod
    def pattern(partial, op, x, kernel_shape):
        return partial(op)(x, kernel_shape=kernel_shape, _outputs=["out"])

    @staticmethod
    def check(context, x, kernel_shape, out):
        return out.producer().attributes.get("strides") is None

    @staticmethod
    def rewrite(partial, op, x, kernel_shape, out):
        attributes = out.producer().attributes

        return partial(op)(
            x, **attributes, strides=len(kernel_shape.as_ints()) * [1]
        )


class RewritePaddedMaxPoolAsPadMaxPool_v1(RewriteRule, Verify):
    """Rewrite padded MaxPool operation as standalone Pad combination."""

    @staticmethod
    def pattern(op, x, pads):
        return op.MaxPool(x, pads=pads, _outputs=["out"])

    @staticmethod
    def check(context, x, pads, out):
        return any(pads.as_ints()) and x.shape is not None

    @staticmethod
    def rewrite(op, x, pads, out):
        # Collect original attributes and delete padding and ceil_mode which are
        # handled by standalone padding.
        attributes = out.producer().attributes
        del attributes["pads"]
        del attributes["ceil_mode"]

        spatial = len(pads := pads.as_ints()) // 2

        return op.MaxPool(
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
            **attributes
        )


@tolerance
class RewritePaddedAveragePoolAsPadAveragePool_v1(RewriteRule, Verify):
    """Rewrite padded AveragePool operation as standalone Pad combination."""

    @staticmethod
    def pattern(op, x, pads, count_include_pad):
        return op.AveragePool(
            x, pads=pads, count_include_pad=count_include_pad, _outputs=["out"]
        )

    @staticmethod
    def check(context, x, pads, count_include_pad, out):
        return any(pads.as_ints()) and x.shape is not None

    @staticmethod
    def rewrite(op, x, pads, count_include_pad, out):
        # Collect original attributes and delete padding and ceil_mode which are
        # handled by standalone padding.
        attributes = out.producer().attributes
        del attributes["pads"]
        del attributes["ceil_mode"]

        spatial = len(pads := pads.as_ints()) // 2

        y = op.AveragePool(
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
            **attributes
        )

        # As standalone padding results in always including the padded elements,
        # apply a correction factor to all border results which now include all
        # the padding zeros in their average.
        if count_include_pad.as_int() == 0:
            del attributes["count_include_pad"]

            # The correction factor is calculated by average pooling over a
            # constant one tensor of the same shape and padding.
            y = op.Div(
                y,
                op.AveragePool(
                    op.CastLike(
                        op.ConstantOfShape(
                            op.Shape(x), value=ir.tensor([1])
                        ),
                        x
                    ),
                    pads=pads,
                    count_include_pad=1,
                    **attributes,
                )
            )

        return y


class RewritePaddedLpPoolAsPadLpPool_v1(RewriteRule, Verify):
    """Rewrite padded LpPool operation as standalone Pad combination."""

    @staticmethod
    def pattern(op, x, pads):
        return op.LpPool(x, pads=pads, _outputs=["out"])

    @staticmethod
    def check(context, x, pads, out):
        return any(pads.as_ints()) and x.shape is not None

    @staticmethod
    def rewrite(op, x, pads, out):
        # Collect original attributes and delete padding and ceil_mode which are
        # handled by standalone padding.
        attributes = out.producer().attributes
        del attributes["pads"]
        del attributes["ceil_mode"]

        spatial = len(pads := pads.as_ints()) // 2

        return op.LpPool(
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
            **attributes
        )
