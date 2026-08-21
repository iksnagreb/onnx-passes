from onnx_passes.passes._base import RewriteRule
from onnx_passes.passes._verify import Verify

import onnx_ir as ir


class MoveAndPastCast_v1(RewriteRule, Verify):
    """Reorder boolean And operation to follow Cast to some numeric type."""

    @staticmethod
    def pattern(op, x, a, to):
        return op.Cast(op.And(x, a), to=to)

    @staticmethod
    def check(context, x, a, to):
        if ir.convenience.get_const_tensor(a) is not None:
            if (dtype := ir.DataType(to.as_int())).is_floating_point():
                return True
            return dtype.is_integer()
        return False

    @staticmethod
    def rewrite(op, x, a, to):
        return op.Mul(op.Cast(x, to=to), op.Cast(a, to=to))

    @property
    def commute(self) -> bool:
        return True


class MoveOrPastCast_v1(RewriteRule, Verify):
    """Reorder boolean Or operation to follow Cast to some numeric type."""

    @staticmethod
    def pattern(op, x, a, to):
        return op.Cast(op.Or(x, a), to=to)

    @staticmethod
    def check(context, x, a, to):
        if ir.convenience.get_const_tensor(a) is not None:
            if (dtype := ir.DataType(to.as_int())).is_floating_point():
                return True
            return dtype.is_integer()
        return False

    @staticmethod
    def rewrite(op, x, a, to):
        return op.Add(
            op.Mul(
                op.Cast(x, to=to),
                op.Cast(
                    op.Where(
                        a,
                        op.Constant(value_float=0.0),
                        op.Constant(value_float=1.0)
                    ),
                    to=to
                )
            ),
            op.Cast(
                op.Where(
                    a,
                    op.Constant(value_float=1.0),
                    op.Constant(value_float=0.0)
                ),
                to=to
            )
        )

    @property
    def commute(self) -> bool:
        return True


class MoveXorPastCast_v1(RewriteRule, Verify):
    """Reorder boolean Xor operation to follow Cast to some numeric type."""

    @staticmethod
    def pattern(op, x, a, to):
        return op.Cast(op.Xor(x, a), to=to)

    @staticmethod
    def check(context, x, a, to):
        if ir.convenience.get_const_tensor(a) is not None:
            if (dtype := ir.DataType(to.as_int())).is_floating_point():
                return True
            return dtype.is_integer() and dtype.is_signed()
        return False

    @staticmethod
    def rewrite(op, x, a, to):
        return op.Add(
            op.Mul(
                op.Cast(x, to=to),
                op.Cast(
                    op.Where(
                        a,
                        op.Constant(value_float=-1.0),
                        op.Constant(value_float=+1.0)
                    ),
                    to=to
                )
            ),
            op.Cast(
                op.Where(
                    a,
                    op.Constant(value_float=+1.0),
                    op.Constant(value_float=+0.0)
                ),
                to=to
            )
        )

    @property
    def commute(self) -> bool:
        return True


class MoveNotPastCast_v1(RewriteRule, Verify):
    """Reorder boolean Not operation to follow Cast to some numeric type."""

    @staticmethod
    def pattern(op, x, to):
        return op.Cast(op.Not(x), to=to)

    @staticmethod
    def check(context, x, to):
        if (dtype := ir.DataType(to.as_int())).is_floating_point():
            return True
        return dtype.is_integer() and dtype.is_signed()

    @staticmethod
    def rewrite(op, x, to):
        return op.Add(
            op.Mul(
                op.Cast(x, to=to),
                op.Cast(op.Constant(value_float=-1.0), to=to)
            ),
            op.Cast(op.Constant(value_float=+1.0), to=to)
        )

    @property
    def commute(self) -> bool:
        return True
