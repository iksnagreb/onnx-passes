from onnx_passes.passes._base import Transformation, Sequential

from onnx_passes.passes._eliminate import cse
from onnx_passes.passes._eliminate import branches


class Eliminate_v1(Sequential, Transformation):
    """Exhaustively applies common eliminations transformations."""

    passes = [
        cse,
        branches
    ]

    exhaustive = True
