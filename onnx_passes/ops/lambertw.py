# Custom operator function registry
from onnx_passes.ops import register, op


# Lambert W function: Approximation via recursive formula according to R. Iacono
# and J.P. Boyd 2017, with starting values according to Lóczi, Lajos 2022.
#
# Note: Only the real-valued primary (k=0) and secondary branch (k=-1) are
#  implemented. For any other value of k this yields NaN.
@register
def LambertW(x, k: int = 0, iterations: int = 4, tolerance: float = 1.0e-8):
    # Define Euler's number as a constant node as it is used frequently down
    # below
    e = op.Exp(op.Constant(value_float=1.0))

    # Select starting values depending on k and x according to Lóczi, Lajos 2022
    w = op.Where(
        # Principal branch of Lambert W with -1/e < x < infinity
        k == 0,
        op.Where(
            e < x,
            op.Log(x) - op.Log(op.Log(x)),
            op.Where(
                op.Constant(value_float=0.0) <= x,
                op.Div(x, e),
                op.Div(
                    e * x * op.Log(1 + op.Sqrt(1 + e * x)),
                    1 + e * x + op.Sqrt(1 + e * x)
                )
            )
        ),
        op.Where(
            # Secondary branch of Lambert W with -1/e < x < 0
            k == -1,
            op.Where(
                op.Constant(value_float=-0.25) < x,
                op.Log(op.Neg(x)) - op.Log(op.Neg(op.Log(op.Neg(x)))),
                -1 - op.Sqrt(2.0) * op.Sqrt(1 + e * x)
            ),
            # Fallback: As ONNX Script needs to trace all branches and cannot
            # raise exceptions, NaN is returned for k not in {0,-1}
            op.Constant(value_float=float("NaN"))
        )
    )

    # Quadratic-rate recursive formula according to R. Iacono and J.P. Boyd 2017
    for i in range(iterations):
        w = (w / (1 + w)) * (1 + op.Log(op.Div(x, w)))

    # Approximation of Lambert W_{k}(x), insert the exact results for values
    # close to the branch point -1/e to avoid numerical issues
    return op.Where(op.Abs(op.Add(x, op.Reciprocal(e))) <= tolerance, -1, w)
