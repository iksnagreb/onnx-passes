# pyright: reportUndefinedVariable=false, reportArgumentType=false, reportReturnType=false
# Use ONNX Script for creating test models
from onnxscript import script, opset18 as op, FLOAT

# Base class/template for deriving pass test cases
from onnx_passes.tests.base import PassesTestBase

from onnx_passes.passes.inline.log_softmax import InlineLogSoftmax

# For generating test inputs
import numpy as np


def _inputs(rank: int):
    values = np.arange(1, 1 + 2 * 3 * 2 * 2, dtype=np.float32) / 10.0
    shape = {1: (3,), 2: (2, 3), 3: (2, 3, 2), 4: (2, 3, 2, 2)}[rank]
    return [values[: np.prod(shape)].reshape(shape)]


def _make_functions(rank: int, axis: int):
    if rank == 1:

        @script(default_opset=op)
        def model(x: FLOAT["C"]) -> FLOAT["C"]:  # type: ignore
            return op.LogSoftmax(x, axis=axis)

        @script(default_opset=op)
        def expected(x: FLOAT["C"]) -> FLOAT["C"]:  # type: ignore
            return op.Log(op.Softmax(x, axis=axis))

        return model, expected

    if rank == 2:

        @script(default_opset=op)
        def model(x: FLOAT["N", "C"]) -> FLOAT["N", "C"]:  # type: ignore
            return op.LogSoftmax(x, axis=axis)

        @script(default_opset=op)
        def expected(x: FLOAT["N", "C"]) -> FLOAT["N", "C"]:  # type: ignore
            return op.Log(op.Softmax(x, axis=axis))

        return model, expected

    if rank == 3:

        @script(default_opset=op)
        def model(x: FLOAT["N", "C", "W"]) -> FLOAT["N", "C", "W"]:  # type: ignore
            return op.LogSoftmax(x, axis=axis)

        @script(default_opset=op)
        def expected(x: FLOAT["N", "C", "W"]) -> FLOAT["N", "C", "W"]:  # type: ignore
            return op.Log(op.Softmax(x, axis=axis))

        return model, expected

    @script(default_opset=op)
    def model(x: FLOAT["N", "C", "H", "W"]) -> FLOAT["N", "C", "H", "W"]:  # type: ignore
        return op.LogSoftmax(x, axis=axis)

    @script(default_opset=op)
    def expected(x: FLOAT["N", "C", "H", "W"]) -> FLOAT["N", "C", "H", "W"]:  # type: ignore
        return op.Log(op.Softmax(x, axis=axis))

    return model, expected


def _axis_tag(axis: int) -> str:
    return f"m{abs(axis)}" if axis < 0 else f"p{axis}"


class _InlineLogSoftmaxTemplate(PassesTestBase):
    __test__ = False
    __passes__ = [InlineLogSoftmax]


def _register_cases() -> None:
    for rank in range(1, 5):
        for axis in range(-rank, rank):
            model, expected = _make_functions(rank, axis)
            name = f"TestInlineLogSoftmaxRank{rank}Axis{_axis_tag(axis)}"
            _InlineLogSoftmaxTemplate.register_case(
                globals(),
                name,
                model=model,
                expected=expected,
                inputs=lambda rank=rank: _inputs(rank),
            )


_register_cases()
