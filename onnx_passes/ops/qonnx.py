# Multi-Threshold function custom operator is implemented using NumPy
import numpy as np

# Registers a python function as implementing a custom ONNX operator
from onnx_passes.ops import register_op, FLOAT, INT64, STRING

# The domain of custom operators exported by QONNX
DOMAIN = "qonnx.custom_op.general"
# Brevitas exports to the brevitas domain, which, however, can be transplated to
# the QONNX domain
BREVITAS_DOMAIN = "onnx.brevitas"


# Resolve rounding modes from string identifiers
ROUNDING_FXS = {
    "ROUND": np.round, "CEIL": np.ceil, "FLOOR": np.floor,
    "ROUND_TO_ZERO": lambda v: np.sign(v) * np.floor(np.abs(v))
}


# QONNX quantizer custom operator implementation to allow models with custom
# quantization to be executed via ONNX Runtime
#   See https://github.com/fastmachinelearning/qonnx for details....
@register_op("Quant", attrs={"signed", "narrow", "rounding_mode"})
def quant(x: FLOAT, scale: FLOAT, zeropoint: FLOAT, bitwidth: FLOAT,
          signed: INT64, narrow: INT64, rounding_mode: STRING):
    # Scale and zero point: Float to Integer
    q = (x / scale) + zeropoint

    # Encode signed 1 bit quantization as bipolar values
    if bitwidth == 1 and signed:
        q = np.where(q >= 0, +1, -1)
    # For all bitwidth larger than 1 clip and round the integer to the range of
    # valid values
    else:
        # Minimum and maximum integer value for the bitwidth, signedness and
        # narrow range combination
        _min = signed * (- 2 ** (bitwidth - signed) + narrow)
        _max = + 2 ** (bitwidth - signed) - 1 - narrow * (1 - signed)
        # Clip the integer to the range and round according tot eh rounding mode
        # while ensuring the data type to stay the same
        q = ROUNDING_FXS[rounding_mode](np.clip(q, _min, _max, dtype=q.dtype))

    # Scale and zero point: Integer to Float
    return (q - zeropoint) * scale
