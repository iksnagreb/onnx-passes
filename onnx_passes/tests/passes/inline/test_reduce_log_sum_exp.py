# Use ONNX Script for creating test models
from onnxscript import script, opset18 as op, FLOAT
import itertools
from typing import Optional, Tuple

# Base class/template for deriving pass test cases
from onnx_passes.tests.base import PassesTestBase

from onnx_passes.passes.inline.reduce_log_sum_exp import InlineReduceLogSumExp

# For generating test inputs
import numpy as np


def _inputs(rank: int):
    values = np.arange(1, 1 + 2 * 3 * 2 * 2, dtype=np.float32) / 10.0
    shape = {1: (3,), 2: (2, 3), 3: (2, 3, 2), 4: (2, 3, 2, 2)}[rank]
    return [values[: np.prod(shape)].reshape(shape)]


def _axes_variants(rank: int):
    variants = [None]
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


def _make_functions(
    rank: int, axes: Optional[Tuple[int, ...]], keepdims: int, noop: int
):
    if axes is None:
        if rank == 1:

            @script(default_opset=op)
            def model(x: FLOAT["C"]) -> FLOAT:
                return op.ReduceLogSumExp(
                    x,
                    keepdims=keepdims,
                    noop_with_empty_axes=noop,
                )

            @script(default_opset=op)
            def expected(x: FLOAT["C"]) -> FLOAT:
                return op.Log(
                    op.ReduceSum(
                        op.Exp(x),
                        keepdims=keepdims,
                        noop_with_empty_axes=noop,
                    )
                )

            return model, expected

        if rank == 2:

            @script(default_opset=op)
            def model(x: FLOAT["N", "C"]) -> FLOAT:
                return op.ReduceLogSumExp(
                    x,
                    keepdims=keepdims,
                    noop_with_empty_axes=noop,
                )

            @script(default_opset=op)
            def expected(x: FLOAT["N", "C"]) -> FLOAT:
                return op.Log(
                    op.ReduceSum(
                        op.Exp(x),
                        keepdims=keepdims,
                        noop_with_empty_axes=noop,
                    )
                )

            return model, expected

        if rank == 3:

            @script(default_opset=op)
            def model(x: FLOAT["N", "C", "W"]) -> FLOAT:
                return op.ReduceLogSumExp(
                    x,
                    keepdims=keepdims,
                    noop_with_empty_axes=noop,
                )

            @script(default_opset=op)
            def expected(x: FLOAT["N", "C", "W"]) -> FLOAT:
                return op.Log(
                    op.ReduceSum(
                        op.Exp(x),
                        keepdims=keepdims,
                        noop_with_empty_axes=noop,
                    )
                )

            return model, expected

        @script(default_opset=op)
        def model(x: FLOAT["N", "C", "H", "W"]) -> FLOAT:
            return op.ReduceLogSumExp(
                x,
                keepdims=keepdims,
                noop_with_empty_axes=noop,
            )

        @script(default_opset=op)
        def expected(x: FLOAT["N", "C", "H", "W"]) -> FLOAT:
            return op.Log(
                op.ReduceSum(
                    op.Exp(x),
                    keepdims=keepdims,
                    noop_with_empty_axes=noop,
                )
            )

        return model, expected

    if rank == 1:

        @script(default_opset=op)
        def model(x: FLOAT["C"]) -> FLOAT:
            return op.ReduceLogSumExp(
                x,
                op.Constant(value_ints=axes),
                keepdims=keepdims,
                noop_with_empty_axes=noop,
            )

        @script(default_opset=op)
        def expected(x: FLOAT["C"]) -> FLOAT:
            return op.Log(
                op.ReduceSum(
                    op.Exp(x),
                    op.Constant(value_ints=axes),
                    keepdims=keepdims,
                    noop_with_empty_axes=noop,
                )
            )

        return model, expected

    if rank == 2:

        @script(default_opset=op)
        def model(x: FLOAT["N", "C"]) -> FLOAT:
            return op.ReduceLogSumExp(
                x,
                op.Constant(value_ints=axes),
                keepdims=keepdims,
                noop_with_empty_axes=noop,
            )

        @script(default_opset=op)
        def expected(x: FLOAT["N", "C"]) -> FLOAT:
            return op.Log(
                op.ReduceSum(
                    op.Exp(x),
                    op.Constant(value_ints=axes),
                    keepdims=keepdims,
                    noop_with_empty_axes=noop,
                )
            )

        return model, expected

    if rank == 3:

        @script(default_opset=op)
        def model(x: FLOAT["N", "C", "W"]) -> FLOAT:
            return op.ReduceLogSumExp(
                x,
                op.Constant(value_ints=axes),
                keepdims=keepdims,
                noop_with_empty_axes=noop,
            )

        @script(default_opset=op)
        def expected(x: FLOAT["N", "C", "W"]) -> FLOAT:
            return op.Log(
                op.ReduceSum(
                    op.Exp(x),
                    op.Constant(value_ints=axes),
                    keepdims=keepdims,
                    noop_with_empty_axes=noop,
                )
            )

        return model, expected

    @script(default_opset=op)
    def model(x: FLOAT["N", "C", "H", "W"]) -> FLOAT:
        return op.ReduceLogSumExp(
            x,
            op.Constant(value_ints=axes),
            keepdims=keepdims,
            noop_with_empty_axes=noop,
        )

    @script(default_opset=op)
    def expected(x: FLOAT["N", "C", "H", "W"]) -> FLOAT:
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
    for rank in range(1, 5):
        for axes in _axes_variants(rank):
            for keepdims in (0, 1):
                for noop in (0, 1):
                    model, expected = _make_functions(rank, axes, keepdims, noop)
                    name = (
                        f"TestInlineReduceLogSumExpRank{rank}"
                        f"{_axes_tag(axes)}"
                        f"Keepdims{keepdims}"
                        f"Noop{noop}"
                    )
                    _InlineReduceLogSumExpTemplate.register_case(
                        globals(),
                        name,
                        model=model,
                        expected=expected,
                        inputs=lambda rank=rank: _inputs(rank),
                    )


_register_cases()
