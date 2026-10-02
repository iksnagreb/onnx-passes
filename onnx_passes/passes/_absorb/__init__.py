from onnx_passes.passes._base import Transformation, Sequential

from onnx_passes.passes._absorb import arithmetic
from onnx_passes.passes._absorb import minmax
from onnx_passes.passes._absorb import exp
from onnx_passes.passes._absorb import log
from onnx_passes.passes._absorb import abs
from onnx_passes.passes._absorb import sqrt
from onnx_passes.passes._absorb import sigmoid
from onnx_passes.passes._absorb import tanh
from onnx_passes.passes._absorb import reciprocal
from onnx_passes.passes._absorb import pow
from onnx_passes.passes._absorb import sign

from onnx_passes.passes import _reorder


class Absorb_v1(Sequential, Transformation):
    """Exhaustively applies common absorption transformations."""

    passes = [
        arithmetic,
        minmax,
        exp,
        log,
        abs,
        sqrt,
        sigmoid,
        tanh,
        reciprocal,
        pow,
        sign,
        _reorder
    ]

    exhaustive = True
