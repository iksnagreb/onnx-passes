from onnx_passes.passes._base import Transformation, Sequential

from onnx_passes.passes._eliminate import cse
from onnx_passes.passes._eliminate import branches
from onnx_passes.passes._eliminate import identity
from onnx_passes.passes._eliminate import annihilator
from onnx_passes.passes._eliminate import idempotence
from onnx_passes.passes._eliminate import involution
from onnx_passes.passes._eliminate import absorption

from onnx_passes.passes import _normalize
from onnx_passes.passes import _fold_constants


class Eliminate_v1(Sequential, Transformation):
    """Exhaustively applies common eliminations transformations."""

    passes = [
        cse,
        branches,
        identity,
        annihilator,
        idempotence,
        involution,
        absorption,
        _fold_constants,
        _normalize,
    ]

    exhaustive = True
