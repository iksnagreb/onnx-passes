from onnx_passes.passes._base import RewriteRuleSetTemplate
from onnx_passes.passes._verify import Verify

from scipy.special import erf

import onnx_ir as ir
import numpy as np


def _t(x):
    """Tanh subexpression in approximate Gelu CDF."""
    return np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3))


def _g0(x):
    """Innermost polynomial subexpression in approximate Gelu CDF."""
    return np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)


def _g1(x):
    """Derivative of _g0 subexpression in approximate Gelu CDF."""
    return np.sqrt(2.0 / np.pi) * (1 + 3 * 0.044715 * x ** 2)


def _g2(x):
    """Second derivative of _g0 subexpression in approximate Gelu CDF."""
    return np.sqrt(2.0 / np.pi) * (6 * 0.044715 * x)


def cdf(x, approximate: str | None = "none"):
    """Cumulative density function used by Gelu with optional approximation."""

    if approximate is not None:
        if approximate == "tanh":
            return 0.5 * (1.0 + _t(x))

    if approximate not in {None, "none"}:
        raise NotImplementedError(
            f"Unsupported {approximate=} for Gelu CDF"
        )

    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def pdf(x, approximate: str | None = "none"):
    """Density function (derivative of the CDF) as used by Gelu."""

    if approximate is not None:
        if approximate == "tanh":
            return 0.5 * (1.0 - _t(x) ** 2) * _g1(x)

    if approximate not in {None, "none"}:
        raise NotImplementedError(
            f"Unsupported {approximate=} for Gelu CDF derivative"
        )

    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-x ** 2 / 2.0)


def pdf_derivative(x, approximate: str | None = "none"):
    """Second derivative of the CDF (first of PDF) as used by Gelu."""

    if approximate is not None:
        if approximate == "tanh":
            return 0.5 * (1.0 - _t(x) ** 2) * (_g2(x) - 2 * _g1(x) ** 2 * _t(x))

    if approximate not in {None, "none"}:
        raise NotImplementedError(
            f"Unsupported {approximate=} for Gelu CDF second derivative"
        )

    return -(1.0 / np.sqrt(2.0 * np.pi)) * x * np.exp(-x ** 2 / 2.0)


def gelu(x, approximate: str | None = "none"):
    """Gelu function with optional approximation according to ONNX."""

    if approximate not in {None, "none", "tanh"}:
        raise NotImplementedError(
            f"Unsupported {approximate=} for Gelu"
        )

    return x * cdf(x, approximate)


def gelu_derivative(x, approximate: str | None = "none"):
    """Derivative of the Gelu function with optional approximation."""

    if approximate not in {None, "none", "tanh"}:
        raise NotImplementedError(
            f"Unsupported {approximate=} for derivative of Gelu"
        )

    return x * pdf(x, approximate) + cdf(x, approximate)


def gelu_second_derivative(x, approximate: str | None = "none"):
    """Second derivative of the Gelu function with optional approximation."""

    if approximate not in {None, "none", "tanh"}:
        raise NotImplementedError(
            f"Unsupported {approximate=} for second derivative of Gelu"
        )

    return x * pdf_derivative(x, approximate) + 2 * pdf(x, approximate)


# Global minimum of the Gelu function: Branchpoint between the decreasing branch
# on the left and the increasing on the right.
#
# Calculated to 64-bit double precision via a fixed-point iteration solving
#   Gelu'(x) = 0 <=> x cdf'(x) + cdf(x) = 0 <=> x = - cdf(x) / cdf'(x),
# selecting the exact option for the cumulative density function and its
# derivative.
GELU_MIN_X: float = -0.75179152469356440
GELU_MIN_Y: float = -0.16997120747990369

# Global minimum of the tanh-approximate Gelu function: Branchpoint between the
# decreasing branch on the left and the increasing on the right.
#
# Calculated to 64-bit double precision via a fixed-point iteration solving
#   Gelu'(x) = 0 <=> x cdf'(x) + cdf(x) = 0 <=> x = - cdf(x) / cdf'(x),
# selecting the approximate option for the cumulative density function and its
# derivative.
GELU_APPROX_MIN_X: float = -0.75242959176670030
GELU_APPROX_MIN_Y: float = -0.16997111967654074


def gelu_inverse(
        y, branch: str, approximate: str | None = "none", niter: int = 100
):
    """Inverse of the Gelu on selected branch with optional approximation."""

    if branch not in {"increasing", "decreasing"}:
        raise NotImplementedError(
            f"Unsupported {branch=} for inverse Gelu"
        )

    if approximate not in {None, "none", "tanh"}:
        raise NotImplementedError(
            f"Unsupported {approximate=} for inverse Gelu"
        )

    # Select a starting point based on the branch to solve Gelu(x) - y = 0 via
    # Halley's method. The branch point is at about -0.75, start at +/- 0.75 to
    # the right/left of this (f' and f'' are non-zero here).
    x = {"increasing": 0, "decreasing": np.where(y > 0, np.nan, -1.5)}[branch]

    # There are no valid solutions below the global minimum of Gelu(x), i.e.,
    # for all x: Gelu(x) >= min_y.
    min_x, min_y = GELU_MIN_X, GELU_MIN_Y

    if approximate is not None:
        if approximate == "tanh":
            min_x, min_y = GELU_APPROX_MIN_X, GELU_APPROX_MIN_Y

    x = np.where(y < min_y, np.nan, np.where(y >= np.inf, np.nan, x))

    # Fixed number of iterations of Halley's method for root finding to solve
    # the equation Gelu(x) = y for x.
    for _ in range(niter):
        # Evaluate f(x) = Gelu(x) - y and its first and second derivative at the
        # current estimate
        f0 = gelu(x, approximate) - y
        f1 = gelu_derivative(x, approximate)
        f2 = gelu_second_derivative(x, approximate)

        # One step of Halley's method sanitized to avoid division of infinity by
        # infinity which can only happen when f is infinite (f' and f'' finite).
        #
        # The only valid solution for this is y = +oo on the increasing branch
        # where x should be +oo as well. In this case f will be -oo.
        inf = np.isinf(f0)

        if branch == "increasing":
            x = np.where(
                inf, np.inf, x - f0 * f1 / np.where(
                    inf, 1.0, (f1 ** 2 - 0.5 * f0 * f2)
                )
            )

        if branch == "decreasing":
            x = np.where(
                inf, np.nan, x - f0 * f1 / (f1 ** 2 - 0.5 * f0 * f2)
            )

        # Also, f' and f'' are never simultaneously zero, thus the denominator
        # can only be zero when f' and f are both zero which happens precisely
        # when x = min_x and y = min_y, for both branches.
        x = np.where(
            (f0 * f1 / (f1 ** 2 - 0.5 * f0 * f2) != 0) | (y != min_y), x, min_x
        )

    return x


class AbsorbGeluIntoComparison_v1(RewriteRuleSetTemplate, Verify):
    """Rewrite comparisons to absorb Gelu into a constant rhs."""

    patterns = (
        lambda op: op.Greater,
        lambda op: op.Less,
        lambda op: op.Equal,
        lambda op: op.GreaterOrEqual,
        lambda op: op.LessOrEqual,
    )

    @staticmethod
    def pattern(partial, op, x, c, approximate):
        return partial(op)(op.Gelu(x, approximate=approximate), c)

    @staticmethod
    def check(context, x, c, approximate):
        return ir.convenience.get_const_tensor(c) is not None

    @staticmethod
    def rewrite(partial, op, x, c, approximate):
        # Branch point and inverse depend on the selected approximation, even
        # though the difference is rather tiny...
        min_x, min_y = GELU_MIN_X, GELU_MIN_Y

        if approximate is not None:
            if (approximate := approximate.as_string()) == "tanh":
                min_x, min_y = GELU_APPROX_MIN_X, GELU_APPROX_MIN_Y

        # Solve Gelu(x) <=> c for both (increasing and decreasing) branches and
        # mask invalid solutions to not propagate NaN into the graph.
        y = ir.convenience.get_const_tensor(c).numpy()

        increasing = gelu_inverse(y, "increasing", approximate)
        decreasing = gelu_inverse(y, "decreasing", approximate)

        valid_increasing = ~np.isnan(increasing)
        valid_decreasing = ~np.isnan(decreasing)

        increasing = np.where(valid_increasing, increasing, 0)
        decreasing = np.where(valid_decreasing, decreasing, 0)

        # Replacement pattern: Up to three solutions of Gelu(x) <=> c, one on
        # each branch and one for c below the global minimum.
        return op.Or(
            op.Or(
                # Solutions on the decreasing branch of Gelu to the left of the
                # global minimum at about (-0.75,-0.16).
                op.And(
                    op.LessOrEqual(
                        x,
                        op.CastLike(
                            op.Constant(value_float=min_x),
                            x
                        )
                    ),
                    op.And(
                        partial(op)(
                            op.CastLike(
                                op.Constant(value=ir.tensor(decreasing)),
                                x
                            ),
                            x
                        ),
                        op.Constant(value=ir.tensor(valid_decreasing))
                    )
                ),
                # Solutions on the increasing branch of Gelu to the right of the
                # global minimum at about (-0.75,-0.16).
                op.And(
                    op.GreaterOrEqual(
                        x,
                        op.CastLike(
                            op.Constant(value_float=min_x),
                            x
                        )
                    ),
                    op.And(
                        partial(op)(
                            x,
                            op.CastLike(
                                op.Constant(value=ir.tensor(increasing)),
                                x
                            )
                        ),
                        op.Constant(value=ir.tensor(valid_increasing))
                    )
                )
            ),
            # For all x: Gelu(x) >= min_y, i.e., Gelu has a global minimum at
            # about (-0.75,-0.16).
            op.And(
                op.Less(
                    c,
                    op.CastLike(
                        op.Constant(value_float=min_y),
                        c
                    )
                ),
                partial(op)(
                    op.CastLike(
                        op.Constant(value_float=min_y),
                        c
                    ),
                    c
                )
            )
        )
