from onnx_passes.passes._base import (
    RewriteRuleSetTemplate, ReplaceWithConstantLike
)
from onnx_passes.passes._verify import Verify

import onnx_ir as ir
import numpy as np


class EliminateNaNComparison_v1(RewriteRuleSetTemplate, Verify):
    """Eliminate comparison to NaN which always is False."""

    patterns = (
        lambda op: op.Equal,
        lambda op: op.Greater,
        lambda op: op.GreaterOrEqual,
        lambda op: op.Less,
        lambda op: op.LessOrEqual
    )

    @staticmethod
    def pattern(partial, op, x, y):
        return partial(op)(x, y)

    @staticmethod
    def check(context, x, y):
        if (x := ir.convenience.get_const_tensor(x)) is not None:
            if np.all(np.isnan(x)):
                return True

        if (y := ir.convenience.get_const_tensor(y)) is not None:
            if np.all(np.isnan(y)):
                return True

        return False

    @staticmethod
    def rewrite(partial, op, x, y):
        return op.Expand(
            op.Cast(
                op.Constant(value_int=0),
                to=ir.DataType.BOOL
            ),
            # Ensure valid shape according to broadcasting
            op.Shape(
                op.Equal(
                    op.ConstantOfShape(op.Shape(x)),
                    op.ConstantOfShape(op.Shape(y))
                )
            )
        )


def match_neg_inf(_, value: ir.Value):
    """Value level checker for matching -inf."""
    if (value := ir.convenience.get_const_tensor(value)) is not None:
        return np.all(value.numpy() == -np.inf)
    return False


def match_pos_inf(_, value: ir.Value):
    """Value level checker for matching +inf."""
    if (value := ir.convenience.get_const_tensor(value)) is not None:
        return np.all(value.numpy() == +np.inf)
    return False


class EliminateNegInfComparisonLhs_v1(ReplaceWithConstantLike, Verify):
    """Eliminate comparison with negative infinity on the left."""

    @staticmethod
    def pattern(op, x):
        return op.Greater(match_neg_inf, x, _outputs=["out"])

    constant = False


class EliminateNegInfComparisonRhs_v1(ReplaceWithConstantLike, Verify):
    """Eliminate comparison with negative infinity on the right."""

    @staticmethod
    def pattern(op, x):
        return op.GreaterOrEqual(x, match_neg_inf, _outputs=["out"])

    constant = True


class EliminatePosInfComparisonLhs_v1(ReplaceWithConstantLike, Verify):
    """Eliminate comparison with positive infinity on the left."""

    @staticmethod
    def pattern(op, x):
        return op.GreaterOrEqual(match_pos_inf, x, _outputs=["out"])

    constant = True


class EliminatePosInfComparisonRhs_v1(ReplaceWithConstantLike, Verify):
    """Eliminate comparison with positive infinity on the right."""

    @staticmethod
    def pattern(op, x):
        return op.Greater(x, match_pos_inf, _outputs=["out"])

    constant = False
