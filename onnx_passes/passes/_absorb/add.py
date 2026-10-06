from onnx_passes.passes._base import RewriteRuleSetTemplate
from onnx_passes.passes._verify import Verify

from onnxscript.rewriter.pattern import OrValue

import onnx_ir as ir


class AbsorbAddIntoComparison_v1(RewriteRuleSetTemplate, Verify):
    """Rewrite comparisons to absorb constant additions.

    Isolates constants on the right hand side of equalities and inequalities by
    adding or subtracting from the other side: e.g., x + a = c <-> x + c - a.
    """

    patterns = (
        lambda op: op.Equal,
        lambda op: op.Greater,
        lambda op: op.GreaterOrEqual,
        lambda op: op.Less,
        lambda op: op.LessOrEqual,
    )

    @staticmethod
    def pattern(partial, op, x, y, z):
        return partial(op)(OrValue([op.Add(x, y), x]), z)

    @staticmethod
    def check(context, x, y, z):
        if ir.convenience.get_const_tensor(x) is not None:
            return True

        if y is not None:
            if ir.convenience.get_const_tensor(y) is not None:
                return True

        if ir.convenience.get_const_tensor(z) is None:
            return True

        return False

    @staticmethod
    def rewrite(partial, op, x, y, z):
        lhs, rhs = None, []

        # Input x when matched as dynamic stays n the lhs or moves as a constant
        # negated on the rhs.
        if (value := ir.convenience.get_const_tensor(x)) is None:
            lhs = x
        else:
            rhs.append(-value.numpy())

        # Input y is optional, only add this if present. Can be either dynamic
        # as matched on the lhs or a constant negated on the rhs.
        if y is not None:
            if (value := ir.convenience.get_const_tensor(y)) is None:
                if lhs:
                    lhs = op.Add(lhs, y)
                else:
                    lhs = y
            else:
                rhs.append(-value.numpy())

        # Input z when matched as dynamic moves to the lhs via subtraction or
        # negation (if it is the only one) or stays as a constant on the rhs.
        if (value := ir.convenience.get_const_tensor(z)) is None:
            if lhs:
                lhs = op.Sub(lhs, z)
            else:
                lhs = op.Neg(z)
        else:
            rhs.append(+value.numpy())

        # Represent comparison to an empty lhs as comparison to zero - should be
        # removed by next constant folding. Inserting the CastLike construction
        # prevents this from being picked up as a constant lhs immediately.
        if not lhs:
            lhs = op.CastLike(op.Constant(value_float=0.0), x)

        # Replacement pattern: Non-constants moved to the left, constants summed
        # up on the right.
        return partial(op)(
            lhs, op.Constant(value=ir.tensor(sum(rhs), dtype=x.dtype))
        )
