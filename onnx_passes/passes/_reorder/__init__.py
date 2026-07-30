from onnx_passes.passes._base import Transformation, Sequential

from onnx_passes.passes._reorder import reshape
from onnx_passes.passes._reorder import transpose

from onnx_passes.passes import _normalize
from onnx_passes.passes import _fold_constants


class Reorder_v1(Sequential, Transformation):
    """Exhaustively applies common reordering transformations."""

    passes = [
        reshape,
        transpose,
        _fold_constants,
        _normalize,
    ]

    exhaustive = True
