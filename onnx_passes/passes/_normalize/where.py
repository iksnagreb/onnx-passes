from onnx_passes.passes._base import RewriteRule
from onnx_passes.passes._verify import Verify

import onnx_ir as ir
import numpy as np


class RewriteComplementTernaryXorAsWhere_v1(RewriteRule, Verify):
    """Rewrite boolean Xor of constant complementary And as Where."""

    @staticmethod
    def pattern(op, a, b, x, y):
        return op.Xor(op.And(a, x), op.And(b, y))

    @staticmethod
    def check(context, a, b, x, y):
        if (a := ir.convenience.get_const_tensor(a)) is not None:
            if (b := ir.convenience.get_const_tensor(b)) is not None:
                return np.all(a.numpy() != b.numpy())
        return False

    @staticmethod
    def rewrite(op, a, b, x, y):
        return op.Where(a, x, y)

    @property
    def commute(self) -> bool:
        return True


class RewriteBooleanWhereAsOr_v1(RewriteRule, Verify):
    """Rewrite Where with all boolean inputs as a boolean Or expression."""

    @staticmethod
    def pattern(op, condition, x, y):
        return op.Where(condition, x, y)

    @staticmethod
    def check(op, condition, x, y):
        return x.dtype == y.dtype == ir.DataType.BOOL

    @staticmethod
    def rewrite(op, condition, x, y):
        return op.Or(op.And(condition, x), op.And(op.Not(condition), y))
