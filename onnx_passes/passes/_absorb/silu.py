from onnx_passes.passes._base import RewriteRule, RewriteRuleSetTemplate
from onnx_passes.passes._verify import Verify

import onnx_ir as ir
import numpy as np


def lambertw(x, k: int = 0, iterations: int = 7, tolerance: float = 1e-17):
    """Lambert W function.

    Approximation via recursive formula according to R. Iacono and J.P. Boyd
    2017, with starting values according to Lóczi, Lajos 2022: these should
    converge to single/double precision within 6/7 iterations.

    Note: Only the real-valued principal (k=0) and secondary branch (k=-1) are
    implemented.
    """

    if k not in {0, -1}:
        raise NotImplementedError(
            f"Unsupported branch {k=} for Lambert W function"
        )

    x = np.asarray(x)
    w = np.asarray(0)

    # Starting values for the principal branch of the Lambert W function with
    # valid inputs x in -1/e < x < oo.
    if k == +0:
        w = np.where(
            # For x in (e,oo), 4/5 iterations to converge in single/double
            # precision
            np.e < x, np.log(x) - np.log(np.log(x)),
            np.where(
                # For x in (0,e), 6/7 iterations to converge to single/double
                # precision
                0 <= x, x / np.e,
                # For x in (-1/e,0), 4/5 iterations to converge to single/double
                # precision
                np.divide(
                    np.e * x * np.log(1 + np.sqrt(1 + np.e * x)),
                    1 + np.e * x + np.sqrt(1 + np.e * x)
                )
            )
        )

    # Starting values for the secondary branch of the Lambert W function with
    # valid inputs x in -1/e < x < 0.
    if k == -1:
        w = np.where(
            # For x in (-1/4,0), 5/6 iterations to converge to single/double
            # precision
            -1 / 4 < x, np.log(-x) - np.log(-np.log(-x)),
            # For x in (-1/e,-1/4), 5/6 iterations to converge to single/double
            # precision
            -1 - np.sqrt(2) * np.sqrt(1 + np.e * x)
        )

    # Approximate Lambert W for x >= -1/e via quadratic-rate recursive formula
    # according to R. Iacono and J.P. Boyd 2017. Configurable iterations, even
    # though for the starting values above, this converges to double precision
    # within 7 iterations.
    for _ in range(iterations):
        w = (w / (1.0 + w)) * (1.0 + np.log(x / w))

    # Approximation of Lambert W, insert the exact results for values close to
    # the branch point -1/e to avoid numerical issues
    return np.where(np.abs(x + 1 / np.e) <= tolerance, -1, w)


def silu(x):
    """Silu activation function: x * Sigmoid(x)."""
    return x / (1 + np.exp(-x))


# Global minimum of the Silu function: Branchpoint between the decreasing branch
# on the left and the increasing on the right.
#
# Calculated to 64-bit double precision via the Lamber W function, solving
#   Silu'(x) = 0 <=> x sigmoid'(x) + sigmoid(x) = 0 <=> x = -1 - W(e^-1)
SILU_MIN_X: float = -1.2784645427610740
SILU_MIN_Y: float = -0.2784645427610738


def silu_inverse(y, branch: str, iterations: int = 7, tolerance: float = 1e-17):
    """Inverse of the Silu function on selected branch: x = y + W(y * e^-y)."""

    if branch not in {"increasing", "decreasing"}:
        raise NotImplementedError(
            f"Unsupported {branch=} for inverse Silu"
        )

    k = {"increasing": 0, "decreasing": -1}[branch]

    return y + lambertw(y * np.exp(-y), k, iterations, tolerance)


class AbsorbSiluIntoComparison_v1(RewriteRuleSetTemplate, Verify):
    """Rewrite comparisons to absorb Silu into a constant rhs."""

    patterns = (
        lambda op: op.Greater,
        lambda op: op.Less,
        lambda op: op.Equal,
        lambda op: op.GreaterOrEqual,
        lambda op: op.LessOrEqual,
    )

    @property
    def commute(self) -> bool:
        return True

    @staticmethod
    def pattern(partial, op, x, c):
        return partial(op)(op.Mul(x, op.Sigmoid(x)), c)

    @staticmethod
    def check(context, x, c):
        return ir.convenience.get_const_tensor(c) is not None

    @staticmethod
    def rewrite(partial, op, x, c):
        # Solve Silu(x) <=> c for both (increasing and decreasing) branches and
        # mask invalid solutions to not propagate NaN into the graph.
        y = ir.convenience.get_const_tensor(c).numpy()

        increasing = silu_inverse(y, "increasing")
        decreasing = silu_inverse(y, "decreasing")

        valid_increasing = ~np.isnan(increasing)
        valid_decreasing = ~np.isnan(decreasing)

        increasing = np.where(valid_increasing, increasing, 0)
        decreasing = np.where(valid_decreasing, decreasing, 0)

        # Replacement pattern: Up to three solutions of Silu(x) <=> c, one on
        # each branch and one for c below the global minimum.
        return op.Or(
            op.Or(
                # Solutions on the decreasing branch of Silu to the left of the
                # global minimum at about (-1.28,-0.28).
                op.And(
                    op.LessOrEqual(
                        x,
                        op.CastLike(
                            op.Constant(value_float=SILU_MIN_X),
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
                # Solutions on the increasing branch of Silu to the right of the
                # global minimum at about (-1.28,-0.28).
                op.And(
                    op.GreaterOrEqual(
                        x,
                        op.CastLike(
                            op.Constant(value_float=SILU_MIN_X),
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
            # For all x: Silu(x) >= min_y, i.e., Silu has a global minimum at
            # about (-1.28,-0.28).
            op.And(
                op.Less(
                    c,
                    op.CastLike(
                        op.Constant(value_float=SILU_MIN_Y),
                        c
                    )
                ),
                partial(op)(
                    op.CastLike(
                        op.Constant(value_float=SILU_MIN_Y),
                        c
                    ),
                    c
                )
            )
        )


class InlineSwish_v1(RewriteRule, Verify):
    """Inline the Swish function (since Opset 24) from its definition."""

    @property
    def commute(self) -> bool:
        return True

    @staticmethod
    def pattern_v24(op, x, alpha):
        return op.Swish(x, alpha=alpha)

    @staticmethod
    def rewrite_v24(op, x, alpha):
        return op.Mul(
            op.Sigmoid(
                op.Mul(
                    op.CastLike(
                        op.Constant(
                            value_float=alpha.as_float()
                        ),
                        x
                    ),
                    x
                )
            ),
            x
        )


class RewriteSwishAsSilu_v1(RewriteRule, Verify):
    """Rewrite the Swish function in terms of Silu for absorption via Silu pass.

    Derived via substitution z = alpha * x and common subexpression extraction
    from the definition of Swish:
        Swish_{alpha}(x) = x * Sigmoid(alpha * x) = Silu(alpha * x) / alpha

    Note: Reordering and elimination passes will always return this back to the
    definition. To prevent this, directly follow with constant folding, CSE and
    multiplication and Silu absorption passes in an exhaustive loop.
    """

    @property
    def commute(self) -> bool:
        return True

    @staticmethod
    def pattern(op, x, alpha):
        return op.Mul(x, op.Sigmoid(op.Mul(alpha, x)))

    @staticmethod
    def rewrite(op, x, alpha):
        return op.Mul(
            op.Mul(
                op.Sigmoid(
                    z := op.Mul(
                        x,
                        alpha
                    )
                ),
                z
            ),
            op.Reciprocal(
                alpha
            )
        )


from onnx_passes.passes._base import Sequential

from onnx_passes.passes import _fold_constants
from onnx_passes.passes._absorb.mul import AbsorbMulIntoComparison_v1


class AbsorbSwishIntoComparison_v1(Sequential):
    """Rewrite comparisons to absorb Swish into a constant rhs.

    Note: This is the exhaustive pass sequence making use of the substitution
    rewriting Swish in terms of Silu absorbed via AbsorbSiluIntoComparison_v1.
    """

    passes = [
        RewriteSwishAsSilu_v1,
        AbsorbMulIntoComparison_v1,
        AbsorbSiluIntoComparison_v1,
        _fold_constants,
    ]

    exhaustive = True
