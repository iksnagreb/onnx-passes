# Custom operator function registry
from onnx_passes.ops import register, op, ir


@register
def Any(x):
    return op.Greater(
        op.ReduceMax(op.Abs(op.Cast(x, to=ir.DataType.INT64)), keepdims=0), 0
    )


@register
def Log2(x):
    return op.Log(x) / op.Log(op.CastLike(op.Constant(value_float=2.0), x))


# Evaluates the unit in the last place (ULP) of floating point inputs x. The ULP
# is the spacing between two consecutive floating-point numbers.
@register
def Ulp(x):
    # Define basic constants matching the type of the input to keep the
    # following more terse and readable
    _0 = op.CastLike(op.Constant(value_float=0.0), x)
    _1 = op.CastLike(op.Constant(value_float=1.0), x)
    _2 = op.CastLike(op.Constant(value_float=2.0), x)

    # Get rid of infinities...
    infinity, x = op.IsInf(x), op.Where(op.IsInf(x), _1, x)

    # Round the input down to the nearest power of two and sanitize zero inputs
    # to avoid taking the log of zero
    x = op.Where(
        x == _0,
        x,
        op.Pow(
            _2, op.Floor(Log2(op.Abs(x) + op.Where(x == _0, _1, _0)))
        )
    )

    # Start searching for the exponent of the Ulp(x) in the middle of the range,
    # expanding to the full shape of the input, as we want the ulp per element
    exponent = op.Expand(_0, op.Shape(x))

    # Increase the ulp exponent while x + ulp == x, this covers all x for which
    # the Ulp(x) is >=1
    condition = Any(x + op.Pow(_2, exponent) == x)
    while condition:
        exponent = exponent + op.Where(x + op.Pow(_2, exponent) == x, _1, _0)
        condition = Any(x + op.Pow(_2, exponent) == x)

    # As the stop condition stops at the exponent where no difference is
    # observed, take back one step to get the Ulp(x)
    exponent = exponent - _1

    # Decrease the ulp exponent while x + ulp > x, this covers all x for which
    # the Ulp(x) is <=1
    condition = Any(x + op.Pow(_2, exponent) > x)
    while condition:
        exponent = exponent - op.Where(x + op.Pow(_2, exponent) > x, _1, _0)
        condition = Any(x + op.Pow(_2, exponent) > x)

    # As the stop condition stops at the exponent where no difference is
    # observed, add back one step to get the Ulp(x)
    return op.Where(infinity, _0, op.Pow(_2, exponent + _1))
