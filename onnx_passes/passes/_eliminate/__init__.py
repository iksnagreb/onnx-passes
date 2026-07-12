from onnx_passes.passes._base import Transformation, Sequential

from onnx_passes.passes._eliminate.cse import CommonSubexpressionElimination_v1


class Eliminate_v1(Sequential, Transformation):
    """Exhaustively applies common eliminations transformations."""

    passes = [
        CommonSubexpressionElimination_v1,
    ]

    exhaustive = True
