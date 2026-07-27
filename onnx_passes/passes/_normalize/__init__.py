from onnx_passes.passes._base import Transformation, Sequential

from onnx_passes.passes._normalize import attributes
from onnx_passes.passes._normalize import reshape
from onnx_passes.passes._normalize import transpose
from onnx_passes.passes._normalize import arithmetic
from onnx_passes.passes._normalize import comparison


class Normalize_v1(Sequential, Transformation):
    """Exhaustively applies common normalization transformations."""

    passes = [
        attributes,
        reshape,
        transpose,
        arithmetic,
        comparison
    ]

    exhaustive = True
