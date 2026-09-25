from onnx_passes.passes._base import Transformation, Sequential

from onnx_passes.passes._absorb import arithmetic
from onnx_passes.passes._absorb import minmax
from onnx_passes.passes._absorb import exp

from onnx_passes.passes import _reorder


class Absorb_v1(Sequential, Transformation):
    """Exhaustively applies common absorption transformations."""

    passes = [
        arithmetic,
        minmax,
        exp,
        _reorder,
    ]

    exhaustive = True
