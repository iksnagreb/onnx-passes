from onnx_passes.passes._base import Transformation, Sequential

from onnx_passes.passes._eliminate import cse
from onnx_passes.passes._eliminate import branches
from onnx_passes.passes._eliminate import identities

from onnx_passes.passes import _normalize
from onnx_passes.passes import _fold_constants


class Eliminate_v1(Sequential, Transformation):
    """Exhaustively applies common eliminations transformations."""

    passes = [
        cse,
        branches,
        identities,
        _fold_constants,
        _normalize,
    ]

    exhaustive = True
