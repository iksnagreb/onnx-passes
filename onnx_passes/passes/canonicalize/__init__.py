from onnx_passes.passes.base import Sequential

from onnx_passes.passes import inline
from onnx_passes.passes import normalize
from onnx_passes.passes import cleanup


class Canonicalize_v1(Sequential):
    """Canonicalize the graph: Combines inlining and normalization passes."""

    passes = [
        cleanup,
        inline,
        normalize
    ]

    exhaustive = True
