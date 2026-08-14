from onnx_passes.passes._base import RewriteRule, RewriteRuleSet
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
        # offers potential for constant elimination, (2) avoid avoerflow when
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


class SimplifyCompoundComparison_v1(RewriteRuleSet, Verify):
    """Simplify compound (And/Or) comparison to a common variable."""

    @staticmethod
    def pattern():
        return [
            # Common variable x on the left side
            lambda op, x, a, b: op.And(
                op.Greater(x, a), op.Greater(x, b)
            ),
            lambda op, x, a, b: op.And(
                op.GreaterOrEqual(x, a), op.GreaterOrEqual(x, b)
            ),
            lambda op, x, a, b: op.Or(
                op.Greater(x, a), op.Greater(x, b)
            ),
            lambda op, x, a, b: op.Or(
                op.GreaterOrEqual(x, a), op.GreaterOrEqual(x, b)
            ),
            # Common variable x on the right side
            lambda op, x, a, b: op.And(
                op.Greater(a, x), op.Greater(b, x)
            ),
            lambda op, x, a, b: op.And(
                op.GreaterOrEqual(a, x), op.GreaterOrEqual(b, x)
            ),
            lambda op, x, a, b: op.Or(
                op.Greater(a, x), op.Greater(b, x)
            ),
            lambda op, x, a, b: op.Or(
                op.GreaterOrEqual(a, x), op.GreaterOrEqual(b, x)
            )
        ]

    @staticmethod
    def rewrite():
        return [
            # Common variable x on the left side
            lambda op, x, a, b: op.Greater(x, op.Max(a, b)),
            lambda op, x, a, b: op.GreaterOrEqual(x, op.Max(a, b)),
            lambda op, x, a, b: op.Greater(x, op.Min(a, b)),
            lambda op, x, a, b: op.GreaterOrEqual(x, op.Min(a, b)),
            # Common variable x on the right side
            lambda op, x, a, b: op.Greater(op.Min(a, b), x),
            lambda op, x, a, b: op.GreaterOrEqual(op.Min(a, b), x),
            lambda op, x, a, b: op.Greater(op.Max(a, b), x),
            lambda op, x, a, b: op.GreaterOrEqual(op.Max(a, b), x)
        ]
