from onnx_passes.passes._base import Transformation, Sequential

from onnx_passes.passes._reorder import reshape


class Reorder_v1(Sequential, Transformation):
    """Exhaustively applies common reordering transformations."""

    passes = [
        reshape,
    ]

    exhaustive = True
