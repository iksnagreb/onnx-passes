from onnx_passes.passes._base import RewriteRule, Sequential, Transformation
from onnx_passes.passes._verify import Verify, tolerance


@tolerance
class RewriteSubAsAdd_v1(RewriteRule, Verify):
    """Rewrite subtraction as addition of the negative."""

    @staticmethod
    def pattern(op, x, y):
        return op.Sub(x, y)

    @staticmethod
    def check(op, x, y):
        return y.dtype is not None and y.dtype.is_signed()

    @staticmethod
    def rewrite(op, x, y):
        return op.Add(x, op.Neg(y))


@tolerance
class RewriteDivAsMul_v1(RewriteRule, Verify):
    """Rewrite division as multiplication of the reciprocal."""

    @staticmethod
    def pattern(op, x, y):
        return op.Div(x, y)

    @staticmethod
    def check(op, x, y):
        return y.dtype is not None and y.dtype.is_floating_point()

    @staticmethod
    def rewrite(op, x, y):
        return op.Mul(x, op.Reciprocal(y))


@tolerance
class RewriteNegAsMul_v1(RewriteRule, Verify):
    """Rewrite negation as multiplication by minus one."""

    @staticmethod
    def pattern(op, x):
        return op.Neg(x)

    @staticmethod
    def rewrite(op, x):
        return op.Mul(op.CastLike(op.Constant(value_int=-1), x), x)
