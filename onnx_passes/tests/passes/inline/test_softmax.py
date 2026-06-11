# pyright: reportArgumentType=false, reportReturnType=false
# Use ONNX Script for creating test models
from onnxscript import script, opset18 as op, FLOAT

# Base class/template for deriving pass test cases
from onnx_passes.tests.base import PassesTestBase, _DIMS_CURRENT

from onnx_passes.passes.inline.softmax import InlineSoftmax

# For generating test inputs
import numpy as np


def _inputs(rank: int):
    values = np.arange(1, 1 + 2 * 3 * 2 * 2, dtype=np.float32) / 10.0
    shape = {1: (3,), 2: (2, 3), 3: (2, 3, 2), 4: (2, 3, 2, 2)}[rank]
    return [values[: np.prod(shape)].reshape(shape)]


def _is_valid_case(rank: int, axis: int) -> bool:
    return -rank <= axis < rank


def _make_functions(rank: int, axis: int):
    @script(default_opset=op)
    def model(x: FLOAT[_DIMS_CURRENT]) -> FLOAT[_DIMS_CURRENT]:
        return op.Softmax(x, axis=axis)

    @script(default_opset=op)
    def expected(x: FLOAT[_DIMS_CURRENT]) -> FLOAT[_DIMS_CURRENT]:
        e = op.Exp(x)
        return op.Div(
            e,
            op.ReduceSum(e, op.Constant(value_ints=[axis]), keepdims=1),
        )

    return model, expected


def _axis_tag(axis: int) -> str:
    return f"m{abs(axis)}" if axis < 0 else f"p{axis}"


class _InlineSoftmaxTemplate(PassesTestBase):
    __test__ = False
    __passes__ = [InlineSoftmax]


def _register_cases() -> None:
    _InlineSoftmaxTemplate.register_sweep_cases(
        globals(),
        sweep={
            "rank": [1, 2, 3, 4],
            "axis": list(range(-4, 4)),
        },
        make_functions=_make_functions,
        name_builder=lambda rank, axis: (
            f"TestInlineSoftmaxRank{rank}Axis{_axis_tag(axis)}"
        ),
        inputs_factory=lambda rank, **_kwargs: _inputs(rank),
        include_case=_is_valid_case,
    )


_register_cases()
