# Multi-Threshold function custom operator is implemented using NumPy
import numpy as np

# Registers a python function as implementing a custom ONNX operator
from onnx_passes.ops import register_op, FLOAT


@register_op(op_type="MultiThreshold")
def multithreshold(x: FLOAT, thresholds: FLOAT, weights: FLOAT) -> FLOAT:
    return np.sum(weights * (x.reshape(*x.shape, 1) >= thresholds), axis=-1)
