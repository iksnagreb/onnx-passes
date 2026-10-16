from onnx_passes.passes.base import Sequential

from onnx_passes.passes.reorder import reshape
from onnx_passes.passes.reorder import expand
from onnx_passes.passes.reorder import slice
from onnx_passes.passes.reorder import split
from onnx_passes.passes.reorder import concat
from onnx_passes.passes.reorder import transpose
from onnx_passes.passes.reorder import commutative
from onnx_passes.passes.reorder import associative
from onnx_passes.passes.reorder import distributive
from onnx_passes.passes.reorder import comparison
from onnx_passes.passes.reorder import arithmetic
from onnx_passes.passes.reorder import reduce
from onnx_passes.passes.reorder import where
from onnx_passes.passes.reorder import im2col

from onnx_passes.passes import normalize
from onnx_passes.passes import fold_constants
from onnx_passes.passes import eliminate


class Reorder_v1(Sequential):
    """Exhaustively apply common reordering transformations."""

    passes = [
        fold_constants,
        normalize,
        eliminate,
        reshape,
        expand,
        slice,
        split,
        concat,
        transpose,
        comparison,
        commutative,
        associative,
        distributive,
        arithmetic,
        reduce,
        where,
        im2col,
    ]

    exhaustive = True
