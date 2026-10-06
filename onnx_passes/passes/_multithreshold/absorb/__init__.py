from onnx_passes.passes._base import Transformation, Sequential

from onnx_passes.passes._multithreshold.absorb import add
from onnx_passes.passes._multithreshold.absorb import mul
from onnx_passes.passes._multithreshold.absorb import minmax
from onnx_passes.passes._multithreshold.absorb import exp
from onnx_passes.passes._multithreshold.absorb import log
from onnx_passes.passes._multithreshold.absorb import abs
from onnx_passes.passes._multithreshold.absorb import sqrt
from onnx_passes.passes._multithreshold.absorb import sigmoid
from onnx_passes.passes._multithreshold.absorb import tanh
from onnx_passes.passes._multithreshold.absorb import reciprocal
from onnx_passes.passes._multithreshold.absorb import pow
from onnx_passes.passes._multithreshold.absorb import sign
from onnx_passes.passes._multithreshold.absorb import gelu
from onnx_passes.passes._multithreshold.absorb import silu

from onnx_passes.passes import _reorder


class Absorb_v1(Sequential, Transformation):
    """Exhaustively applies common absorption transformations."""

    passes = [
        add,
        mul,
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
        gelu,
        silu,
        _reorder
    ]

    exhaustive = True
