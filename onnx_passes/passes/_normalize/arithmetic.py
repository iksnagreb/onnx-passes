from onnx_passes.passes._base import RewriteRule
from onnx_passes.passes._verify import Verify, tolerance

from onnxscript.rewriter.pattern import OrValue


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


@tolerance
class RewriteMulAsPow_v1(RewriteRule, Verify):
    """Rewrite multiplication (x ** y) * (x ** z) as power x ** (y + z).

    The matched powers y and z are optional, i.e., implicitly 1 so x * x is
    rewritten as x ** 2.
    """

    @staticmethod
    def pattern(op, x, y, z):
        return op.Mul(
            OrValue([op.Pow(x, y), x]),
            OrValue([op.Pow(x, z), x])
        )

    @staticmethod
    def rewrite(op, x, y, z):
        if y is not None:
            if z is not None:
                return op.Pow(
                    x, op.Add(y, z)
                )

            return op.Pow(
                x, op.Add(y, op.CastLike(op.Constant(value_float=1.0), y))
            )

        if z is not None:
            return op.Pow(
                x, op.Add(z, op.CastLike(op.Constant(value_float=1.0), z))
            )

        return op.Pow(x, op.CastLike(op.Constant(value_float=2.0), x))


@tolerance
class FuseConsecutivePows_v1(RewriteRule, Verify):
    """Fuses two consecutive power operations into a single power."""

    @staticmethod
    def pattern(op, x, y, z):
        return op.Pow(op.Pow(x, y), z)

    @staticmethod
    def rewrite(op, x, y, z):
        return op.Pow(x, op.Mul(y, z))


@tolerance
class UnrollSum_v1(RewriteRule, Verify):
    """Rewrite multi-input Sum as a tree/chain of binary Add operations."""

    @staticmethod
    def pattern(op, x):
        return op.Sum(x, _allow_other_inputs=True, _outputs=["out"])

    @staticmethod
    def rewrite(op, x, out):
        for value in out.producer().inputs[1:]:
            x = op.Add(x, value)

        return x
