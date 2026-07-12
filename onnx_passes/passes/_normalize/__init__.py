from onnx_passes.passes._base import Transformation, Sequential

from onnx_passes.passes._normalize import attributes
from onnx_passes.passes._normalize import reshape


class Normalize_v1(Sequential, Transformation):
    """Exhaustively applies common normalization transformations."""

    passes = [
        attributes,
        reshape
    ]

    exhaustive = True
