from onnx_passes.passes._base import (
    RewriteRuleSetTemplate, Sequential, Transformation
)
from onnx_passes.passes._verify import Verify

import onnx_ir as ir


class InlineTanhIntoComparison_v1(RewriteRuleSetTemplate, Verify):
    """Rewrite comparisons to inline and partially solve Tanh for x on lhs.

    Derived by inlining the definition
    Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)), and simplifying until the
    (in-)equality would split into cases.
    """

    patterns = (
        lambda op: op.Greater,
        lambda op: op.Less,
        lambda op: op.Equal,
        lambda op: op.GreaterOrEqual,
        lambda op: op.LessOrEqual,
    )

    @staticmethod
    def pattern(partial, op, x, c):
        return partial(op)(op.Tanh(x), c)

    @staticmethod
    def check(context, x, c):
        return ir.convenience.get_const_tensor(c) is not None

    @staticmethod
    def rewrite(partial, op, x, c):
        return partial(op)(
            op.Mul(
                op.Sub(
                    op.CastLike(
                        op.Constant(value_float=1.0),
                        c
                    ),
                    c
                ),
                op.Exp(
                    op.Add(x, x)
                )
            ),
            op.Add(
                op.CastLike(
                    op.Constant(value_float=1.0),
                    c
                ),
                c
            ),
        )


from onnx_passes.passes._normalize.arithmetic import RewriteNegAsMul_v1
from onnx_passes.passes._normalize.arithmetic import RewriteSubAsAdd_v1

from onnx_passes.passes._absorb.arithmetic import AbsorbMulIntoComparison_v1
from onnx_passes.passes._absorb.arithmetic import AbsorbAddIntoComparison_v1

from onnx_passes.passes._absorb.exp import AbsorbExpIntoComparison_v1

from onnx_passes.passes import _fold_constants


class AbsorbTanhIntoComparison_v1(Sequential, Transformation):
    """Rewrite comparisons to absorb Tanh into constant rhs."""

    passes = [
        InlineTanhIntoComparison_v1,
        RewriteNegAsMul_v1,
        RewriteSubAsAdd_v1,
        _fold_constants,
        AbsorbMulIntoComparison_v1,
        AbsorbAddIntoComparison_v1,
        AbsorbExpIntoComparison_v1,
    ]

    exhaustive = True
