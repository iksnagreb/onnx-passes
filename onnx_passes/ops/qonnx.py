# Quantizer custom operator is implemented using NumPy
import numpy as np

# Implementing and registering custom operators with the ai.onnx.contrib domain
from onnxruntime_extensions import onnx_op, PyOp

# The domain of custom operators exported by QONNX
DOMAIN = "qonnx.custom_op.general"
# Brevitas exports to the brevitas domain, which, however, can be transplanted
# to the QONNX domain
BREVITAS_DOMAIN = "onnx.brevitas"

# Resolve rounding modes from string identifiers
ROUNDING_FXS = {
    "ROUND": np.round, "CEIL": np.ceil, "FLOOR": np.floor,
    "ROUND_TO_ZERO": lambda v: np.sign(v) * np.floor(np.abs(v))
}


# QONNX quantizer custom operator implementation to allow models with custom
# quantization to be executed via ONNX Runtime
#   See https://github.com/fastmachinelearning/qonnx for details....
@onnx_op(
    op_type="Quant",
    inputs=[
        PyOp.dt_float,  # x
        PyOp.dt_float,  # scale
        PyOp.dt_float,  # zeropoint
        PyOp.dt_float,  # bitwidth
    ],
    outputs=[PyOp.dt_float],
    attrs={
        "signed": PyOp.dt_int64,
        "narrow": PyOp.dt_int64,
        "rounding_mode": PyOp.dt_string
    }
)
def quant(x, scale, zeropoint, bitwidth, signed, narrow, rounding_mode):
    # Scale and zero point: Float to Integer
    q = (x / scale) + zeropoint  # noqa: Duplicate of .passes.inline.qonnx

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
