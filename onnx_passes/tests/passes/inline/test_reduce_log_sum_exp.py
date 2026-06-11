# pyright: reportArgumentType=false, reportReturnType=false
# Use ONNX Script for creating test models
from onnxscript import script, opset18 as op, FLOAT
import itertools
from typing import Optional, Tuple

# Base class/template for deriving pass test cases
from onnx_passes.tests.base import PassesTestBase, _DIMS_CURRENT

from onnx_passes.passes.inline.reduce_log_sum_exp import InlineReduceLogSumExp

# For generating test inputs
import numpy as np


def _inputs(rank: int):
    values = np.arange(1, 1 + 2 * 3 * 2 * 2, dtype=np.float32) / 10.0
    shape = {1: (3,), 2: (2, 3), 3: (2, 3, 2), 4: (2, 3, 2, 2)}[rank]
    return [values[: np.prod(shape)].reshape(shape)]


def _axes_variants(rank: int):
    variants: list[Optional[Tuple[int, ...]]] = [None]
    seen = set()

    for n in range(rank + 1):
        for combo in itertools.combinations(range(rank), n):
            options = [tuple(combo)]
            if combo:
                options.append(tuple(i - rank for i in combo))

            for axes in options:
                if axes not in seen:
                    seen.add(axes)
                    variants.append(axes)

    return variants


def _is_valid_case(rank: int, axes: Optional[Tuple[int, ...]], **_kwargs) -> bool:
    if axes is None:
        return True
    return all(-rank <= axis < rank for axis in axes)


def _make_functions(
    rank: int, axes: Optional[Tuple[int, ...]], keepdims: int, noop: int
):
    if axes is None:

        @script(default_opset=op)
        def model(x: FLOAT[_DIMS_CURRENT]) -> FLOAT:
            return op.ReduceLogSumExp(
                x,
                keepdims=keepdims,
                noop_with_empty_axes=noop,
            )

        @script(default_opset=op)
        def expected(x: FLOAT[_DIMS_CURRENT]) -> FLOAT:
            return op.Log(
                op.ReduceSum(
                    op.Exp(x),
                    keepdims=keepdims,
                    noop_with_empty_axes=noop,
                )
            )

        return model, expected

    @script(default_opset=op)
    def model(x: FLOAT[_DIMS_CURRENT]) -> FLOAT:
        return op.ReduceLogSumExp(
            x,
            op.Constant(value_ints=axes),
            keepdims=keepdims,
            noop_with_empty_axes=noop,
        )

    @script(default_opset=op)
    def expected(x: FLOAT[_DIMS_CURRENT]) -> FLOAT:
        return op.Log(
            op.ReduceSum(
                op.Exp(x),
                op.Constant(value_ints=axes),
                keepdims=keepdims,
                noop_with_empty_axes=noop,
            )
        )

    return model, expected


def _axis_tag(axis: int) -> str:
    return f"m{abs(axis)}" if axis < 0 else f"p{axis}"


def _axes_tag(axes: Optional[Tuple[int, ...]]) -> str:
    if axes is None:
        return "NoAxes"
    if len(axes) == 0:
        return "EmptyAxes"
    return "Axes" + "_".join(_axis_tag(axis) for axis in axes)


class _InlineReduceLogSumExpTemplate(PassesTestBase):
    __test__ = False
    __passes__ = [InlineReduceLogSumExp]


def _register_cases() -> None:
    _InlineReduceLogSumExpTemplate.register_sweep_cases(
        globals(),
        sweep={
            "rank": [1, 2, 3, 4],
            "axes": _axes_variants(4),
            "keepdims": [0, 1],
            "noop": [0, 1],
        },
        make_functions=_make_functions,
        name_builder=lambda rank, axes, keepdims, noop: (
            f"TestInlineReduceLogSumExpRank{rank}"
            f"{_axes_tag(axes)}"
            f"Keepdims{keepdims}"
            f"Noop{noop}"
        ),
        inputs_factory=lambda rank, **_kwargs: _inputs(rank),
        include_case=_is_valid_case,
    )


_register_cases()
