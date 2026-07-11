from onnx_passes.passes._base import Sequential, Transformation

from onnx_passes.passes import _inline, _normalize


class Canonicalize_v1(Sequential, Transformation):
    """Canonicalize the graph: Combines inlining and normalization passes."""
    passes = [
        _inline, _normalize
    ]
