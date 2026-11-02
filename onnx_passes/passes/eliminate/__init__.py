from onnx_passes.passes.base import Sequential

from onnx_passes.passes.eliminate import cse
from onnx_passes.passes.eliminate import branches
from onnx_passes.passes.eliminate import identity
from onnx_passes.passes.eliminate import annihilator
from onnx_passes.passes.eliminate import idempotence
from onnx_passes.passes.eliminate import involution
from onnx_passes.passes.eliminate import absorption
from onnx_passes.passes.eliminate import inverse
from onnx_passes.passes.eliminate import complement
from onnx_passes.passes.eliminate import comparison

from onnx_passes.passes import normalize
from onnx_passes.passes import fold_constants


class Eliminate_v1(Sequential):
    """Exhaustively applies common eliminations transformations."""

    passes = [
        cse,
        branches,
        identity,
        annihilator,
        idempotence,
        involution,
        absorption,
        inverse,
        complement,
        comparison,
        fold_constants,
        normalize,
    ]

    exhaustive = True
