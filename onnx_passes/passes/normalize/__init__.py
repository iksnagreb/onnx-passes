from onnx_passes.passes.base import Sequential

from onnx_passes.passes.normalize import attributes
from onnx_passes.passes.normalize import reshape
from onnx_passes.passes.normalize import slice
from onnx_passes.passes.normalize import transpose
from onnx_passes.passes.normalize import conv
from onnx_passes.passes.normalize import pooling
from onnx_passes.passes.normalize import pad
from onnx_passes.passes.normalize import arithmetic
from onnx_passes.passes.normalize import comparison
from onnx_passes.passes.normalize import minmax
from onnx_passes.passes.normalize import where
from onnx_passes.passes.normalize import reduce


class Normalize_v1(Sequential):
    """Exhaustively applies common normalization transformations."""

    passes = [
        attributes,
        reshape,
        slice,
        transpose,
        conv,
        pooling,
        pad,
        arithmetic,
        comparison,
        minmax,
        where,
        reduce,
    ]

    exhaustive = True
