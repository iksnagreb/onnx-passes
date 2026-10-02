from onnx_passes.passes._base import RewriteRule
from onnx_passes.passes._verify import Verify

import onnx_ir as ir


class RewriteBooleanWhereAsXor_v1(RewriteRule, Verify):
    """Rewrite Where with all boolean inputs as a boolean Xor expression."""

    @staticmethod
    def pattern(op, condition, x, y):
        return op.Where(condition, x, y)

    @staticmethod
    def check(op, condition, x, y):
        return x.dtype == y.dtype == ir.DataType.BOOL

    @staticmethod
    def rewrite(op, condition, x, y):
        return op.Xor(op.And(condition, x), op.And(op.Not(condition), y))
