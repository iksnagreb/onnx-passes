from onnx_passes.passes._base import Transformation, Sequential

from onnx_passes.passes._reorder import reshape
from onnx_passes.passes._reorder import slice
from onnx_passes.passes._reorder import split
from onnx_passes.passes._reorder import concat
from onnx_passes.passes._reorder import transpose
from onnx_passes.passes._reorder import commutative
from onnx_passes.passes._reorder import associative
from onnx_passes.passes._reorder import distributive

from onnx_passes.passes import _normalize
from onnx_passes.passes import _fold_constants
from onnx_passes.passes import _eliminate


class Reorder_v1(Sequential, Transformation):
    """Exhaustively applies common reordering transformations."""

    passes = [
        reshape,
        slice,
        split,
        concat,
        transpose,
        commutative,
        associative,
        distributive,
        _fold_constants,
        _normalize,
        _eliminate
    ]

    exhaustive = True
