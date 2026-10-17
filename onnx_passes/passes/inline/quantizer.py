from onnx_passes.passes.base import Sequential, RewriteRule
from onnx_passes.passes.verify import Verify

from onnx_passes.ops import DOMAIN as PASSES_DOMAIN
from onnx_passes.ops.qonnx import DOMAIN as QONNX_DOMAIN

import onnx_ir as ir
import numpy as np


class InlineQuant_v1(RewriteRule, Verify):
    """Inline ONNX Passes custom quantizer into the graph."""

    @staticmethod
    def pattern(op, x, s, z, bitwidth, signed, narrow):
        return op.Quant(
            x,
            s,
            z,
            bitwidth,
            signed=signed,
            narrow=narrow,
            _domain=PASSES_DOMAIN
        )

    @staticmethod
    def rewrite(op, x, s, z, bitwidth, signed, narrow):
        # Convert signed and narrow mode attributes to tensors matching the
        # input type to be used in dynamic calculations.
        signed = op.CastLike(
            op.Constant(value_int=signed.as_int()),
            x
        )

        narrow = op.CastLike(
            op.Constant(value_int=narrow.as_int()),
            x
        )

        # Minimum clipping range of the integer quantization:
        #   (-2 ** (bitwidth - signed) + narrow) * signed
        minimum = op.Mul(
            op.Add(
                op.Neg(
                    op.Pow(
                        op.CastLike(
                            op.Constant(value_float=2.0),
                            x
                        ),
                        op.Sub(
                            op.CastLike(
                                bitwidth,
                                x
                            ),
                            signed
                        )
                    )
                ),
                narrow
            ),
            signed
        )

        # Maximum clipping range of the integer quantization:
        #   2 ** (bitwidth - signed) - 1 - narrow * (1 - signed)
        maximum = op.Sub(
            op.Sub(
                op.Pow(
                    op.CastLike(
                        op.Constant(value_float=2.0),
                        x
                    ),
                    op.Sub(
                        op.CastLike(
                            bitwidth,
                            x
                        ),
                        signed
                    )
                ),
                op.Mul(
                    narrow,
                    op.Sub(
                        op.CastLike(
                            op.Constant(value_float=1.0),
                            x
                        ),
                        signed
                    )
                )
            ),
            op.CastLike(
                op.Constant(value_float=1.0),
                x
            )
        )

        # Divide input by scale and shift by zeropoint: Float to integer
        # quantization, still with fractional part and can be out of range.
        q = op.Add(op.Div(x, s), z)

        # Clipping and rounding according to the selected rounding mode or
        # special treatment for signed 1-bit quantization.
        q = op.Where(
            # Signed 1-bit quantization: Convert to bipolar encoding
            # where q >= 0 is encoded as +1 and q < 0 as -1
            op.And(
                op.Equal(
                    bitwidth,
                    op.CastLike(
                        op.Constant(value_int=1),
                        bitwidth
                    )
                ),
                op.Cast(
                    signed, to=ir.DataType.BOOL
                )
            ),
            op.Where(
                op.GreaterOrEqual(
                    q,
                    op.CastLike(
                        op.Constant(value_float=0.0),
                        q
                    )
                ),
                op.CastLike(
                    op.Constant(value_float=+1.0),
                    q
                ),
                op.CastLike(
                    op.Constant(value_float=-1.0),
                    q
                )
            ),
            # Non-bipolar encodings: Clip to integer bounds and round
            # according to the standard ONNX rounding mode
            op.Round(op.Clip(q, minimum, maximum))
        )

        # Multiply quantized value by scale and shift by zeropoint: Integer
        # to float dequantization.
        return op.Mul(
            op.Sub(
                # If this quantizer is on a constant/parameter path, block the
                # flow of constant-folding at the quantized values.
                op.ConstantStop(
                    q,
                    _domain=PASSES_DOMAIN
                ),
                z
            ),
            s
        )


class InlineQONNXQuant_v1(RewriteRule, Verify):
    """Inline QONNX custom quantizer into the graph."""

    @staticmethod
    def pattern(op, x, s, z, bitwidth, signed, narrow, rounding_mode):
        return op.Quant(
            x,
            s,
            z,
            bitwidth,
            signed=signed,
            narrow=narrow,
            rounding_mode=rounding_mode,
            _domain=QONNX_DOMAIN
        )

    @staticmethod
    def rewrite(op, x, s, z, bitwidth, signed, narrow, rounding_mode):
        # Convert signed and narrow mode attributes to tensors matching the
        # input type to be used in dynamic calculations.
        signed = op.CastLike(
            op.Constant(value_int=signed.as_int()),
            x
        )

        narrow = op.CastLike(
            op.Constant(value_int=narrow.as_int()),
            x
        )

        # Minimum clipping range of the integer quantization:
        #   (-2 ** (bitwidth - signed) + narrow) * signed
        minimum = op.Mul(
            op.Add(
                op.Neg(
                    op.Pow(
                        op.CastLike(
                            op.Constant(value_float=2.0),
                            x
                        ),
                        op.Sub(
                            op.CastLike(
                                bitwidth,
                                x
                            ),
                            signed
                        )
                    )
                ),
                narrow
            ),
            signed
        )

        # Maximum clipping range of the integer quantization:
        #   2 ** (bitwidth - signed) - 1 - narrow * (1 - signed)
        maximum = op.Sub(
            op.Sub(
                op.Pow(
                    op.CastLike(
                        op.Constant(value_float=2.0),
                        x
                    ),
                    op.Sub(
                        op.CastLike(
                            bitwidth,
                            x
                        ),
                        signed
                    )
                ),
                op.Mul(
                    narrow,
                    op.Sub(
                        op.CastLike(
                            op.Constant(value_float=1.0),
                            x
                        ),
                        signed
                    )
                )
            ),
            op.CastLike(
                op.Constant(value_float=1.0),
                x
            )
        )

        # Divide input by scale and shift by zeropoint: Float to integer
        # quantization, still with fractional part and can be out of range.
        q = op.Add(op.Div(x, s), z)

        # Clipping to the integer range, used in all rounding mode branches
        # below - avoid inserting this into all if-then subgraphs.
        c = op.Clip(q, minimum, maximum)

        # Rounding mode "ROUND" - round to even, matches standard ONNX Round
        # operator behavior
        if rounding_mode.as_string() == "ROUND":
            y = op.Round(c)

        # Rounding mode "CEIL" -  matches standard ONNX Ceil operator
        # behavior
        elif rounding_mode.as_string() == "CEIL":
            y = op.Ceil(c)

        # Rounding mode "FLOOR" - matches standard ONNX Ceil operator
        # behavior
        elif rounding_mode.as_string() == "FLOOR":
            y = op.Floor(c)

        # Rounding mode "UP" - combination of standard ONNX operator
        # behaviors
        elif rounding_mode.as_string() == "UP":
            y = op.Mul(
                op.Ceil(
                    op.Abs(
                        c
                    )
                ),
                op.Sign(
                    c
                )
            )

        # Rounding mode "DOWN" - combination of standard ONNX operator
        # behaviors
        elif rounding_mode.as_string() == "DOWN":
            y = op.Mul(
                op.Floor(
                    op.Abs(
                        c
                    )
                ),
                op.Sign(
                    c
                )
            )

        # Rounding mode "HALF_UP" - combination of standard ONNX operator
        # behaviors
        elif rounding_mode.as_string() == "HALF_UP":
            y = op.Mul(
                op.Add(
                    op.Floor(
                        op.Abs(
                            c
                        )
                    ),
                    op.CastLike(
                        op.Constant(
                            value_float=0.5
                        ),
                        c
                    )
                ),
                op.Sign(
                    c
                )
            )

        # Rounding mode "HALF_DOWN" - combination of standard ONNX operator
        # behaviors
        elif rounding_mode.as_string() == "HALF_DOWN":
            y = op.Mul(
                op.Sub(
                    op.Ceil(
                        op.Abs(
                            c
                        )
                    ),
                    op.CastLike(
                        op.Constant(
                            value_float=0.5
                        ),
                        c
                    )
                ),
                op.Sign(
                    c
                )
            )

        # Else branch cannot be omitted, and it is not possible to raise
        # exceptions or assertions - fallback to NaN...
        else:
            y = op.CastLike(
                op.Constant(value_float=np.nan),
                c
            )

        # Clipping and rounding according to the selected rounding mode or
        # special treatment for signed 1-bit quantization.
        q = op.Where(
            # Signed 1-bit quantization: Convert to bipolar encoding
            # where q >= 0 is encoded as +1 and q < 0 as -1
            op.And(
                op.Equal(
                    bitwidth,
                    op.CastLike(
                        op.Constant(value_int=1),
                        bitwidth
                    )
                ),
                op.Cast(
                    signed, to=ir.DataType.BOOL
                )
            ),
            op.Where(
                op.GreaterOrEqual(
                    q,
                    op.CastLike(
                        op.Constant(value_float=0.0),
                        q
                    )
                ),
                op.CastLike(
                    op.Constant(value_float=+1.0),
                    q
                ),
                op.CastLike(
                    op.Constant(value_float=-1.0),
                    q
                )
            ),
            # Non-bipolar encodings: Clip to integer bounds and round
            # according to the rounding mode
            y
        )

        # Multiply quantized value by scale and shift by zeropoint: Integer
        # to float dequantization.
        return op.Mul(
            op.Sub(
                # If this quantizer is on a constant/parameter path, block the
                # flow of constant-folding at the quantized values.
                op.ConstantStop(
                    q,
                    _domain=PASSES_DOMAIN
                ),
                z
            ),
            s
        )


class InlineQuantizer_v1(Sequential):
    """Apply quantizer inlining transformations."""

    passes = [
        InlineQuant_v1,
        InlineQONNXQuant_v1,
    ]
