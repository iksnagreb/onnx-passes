from onnx_passes.passes._base import (
    RewriteRule, RewriteRuleSet, RewriteRuleSetTemplate
)
from onnx_passes.passes._verify import Verify

import onnx_ir as ir
import numpy as np


class RewriteLessAsGreater_v1(RewriteRuleSet, Verify):
    """Rewrite Less than comparison as Greater than comparison."""

    @staticmethod
    def pattern():
        return [
            lambda op, x, y: op.Less(x, y),
            lambda op, x, y: op.LessOrEqual(x, y),
        ]

    @staticmethod
    def rewrite():
        return [
            lambda op, x, y: op.Greater(y, x),
            lambda op, x, y: op.GreaterOrEqual(y, x),
        ]


class RewriteGreaterAsGreaterOrEqual_v1(RewriteRule, Verify):
    """Rewrite strict as non-strict greater than comparison to constants."""

    @staticmethod
    def pattern(op, x, c):
        return op.Greater(x, c)

    @staticmethod
    def check(context, x, c):
        return ir.convenience.get_const_tensor(c) is not None

    @staticmethod
    def rewrite(op, x, c: ir.Value):
        if c.dtype.is_integer():  # noqa: dtype is never None
            return op.And(
                # Non-strict comparison to next larger integer
                op.GreaterOrEqual(
                    x,
                    op.Add(
                        c,
                        op.CastLike(
                            op.Constant(value_int=1),
                            c
                        )
                    )
                ),
                # Mask: x > dtype.max is always False
                op.Not(
                    op.Equal(
                        c,
                        op.CastLike(
                            op.Constant(value_int=c.dtype.max),  # noqa: dtype
                            c
                        )
                    )
                )
            )

        # Sanitized nextafter calculations in NumPy: (1) wrap around infinity
        # to negative infinity as this is masked to False anyway but x >= -inf
        # offers potential for constant elimination, (2) avoid overflow when
        # applying nextafter to the maximum non infinity value, and (3) apply
        # nextafter to all other valid inputs.
        y = ir.convenience.get_const_tensor(c).numpy().copy()  # noqa: not None

        is_inf = y == np.inf
        is_max = y == c.dtype.max  # noqa: dtype
        finite = ~is_inf & ~is_max

        y[is_inf] = np.asarray(-np.inf, dtype=y.dtype)
        y[is_max] = np.asarray(+np.inf, dtype=y.dtype)  # noqa: dtype
        y[finite] = np.nextafter(y[finite], np.asarray(np.inf, dtype=y.dtype))

        return op.And(
            # Non-strict comparison to next larger float
            op.GreaterOrEqual(
                x,
                op.CastLike(
                    op.Constant(value=ir.tensor(y)),
                    c
                )
            ),
            # Mask: x > inf is always False
            op.Not(
                op.Equal(
                    c,
                    op.CastLike(
                        op.Constant(value_float=np.inf),
                        c
                    )
                )
            )
        )


class SimplifyCompoundComparisonLhs_v1(RewriteRuleSetTemplate, Verify):
    """Simplify compound (And/Or) comparison to a common variable on the lhs."""

    patterns = (
        # x < a & x < b -> x < Min(a,b)
        lambda op: (op.Less, op.And, op.Min),
        lambda op: (op.LessOrEqual, op.And, op.Min),
        # x > a & x > b -> x > Max(a,b)
        lambda op: (op.Greater, op.And, op.Max),
        lambda op: (op.GreaterOrEqual, op.And, op.Max),
        # x < a | x < b -> x < Max(a,b)
        lambda op: (op.Less, op.Or, op.Max),
        lambda op: (op.LessOrEqual, op.Or, op.Max),
        # x > a | x > b -> x > Min(a,b)
        lambda op: (op.Greater, op.Or, op.Min),
        lambda op: (op.GreaterOrEqual, op.Or, op.Min),
    )

    @staticmethod
    def pattern(partial, op, x, a, b):
        return partial(op)[1](partial(op)[0](x, a), partial(op)[0](x, b))

    @staticmethod
    def check(context, x, a, b):
        if ir.convenience.get_const_tensor(a) is not None:
            if ir.convenience.get_const_tensor(b) is not None:
                return ir.convenience.get_const_tensor(x) is None

        return False

    @staticmethod
    def rewrite(partial, op, x, a, b):
        return partial(op)[0](x, partial(op)[2](a, b))


class SimplifyCompoundComparisonRhs_v1(RewriteRuleSetTemplate, Verify):
    """Simplify compound (And/Or) comparison to a common variable on the rhs."""

    patterns = (
        # a < x & b < x -> Max(a,b) < x
        lambda op: (op.Less, op.And, op.Max),
        lambda op: (op.LessOrEqual, op.And, op.Max),
        # a > x & b > x -> Min(a,b) > x
        lambda op: (op.Greater, op.And, op.Min),
        lambda op: (op.GreaterOrEqual, op.And, op.Min),
        # a < x | b < x -> Min(a,b) < x
        lambda op: (op.Less, op.Or, op.Min),
        lambda op: (op.LessOrEqual, op.Or, op.Min),
        # a > x | b > x -> Max(a,b) > x
        lambda op: (op.Greater, op.Or, op.Max),
        lambda op: (op.GreaterOrEqual, op.Or, op.Max),
    )

    @staticmethod
    def pattern(partial, op, x, a, b):
        return partial(op)[1](partial(op)[0](a, x), partial(op)[0](b, x))

    @staticmethod
    def check(context, x, a, b):
        if ir.convenience.get_const_tensor(a) is not None:
            if ir.convenience.get_const_tensor(b) is not None:
                return ir.convenience.get_const_tensor(x) is None

        return False

    @staticmethod
    def rewrite(partial, op, x, a, b):
        return partial(op)[0](partial(op)[2](a, b), x)
