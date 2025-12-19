# Custom operator function registry
from onnx_passes.ops import register, op

# Inverse Swish function is expressed in terms to the Lambert W function and
# uses the Swish to find bounding values
from onnx_passes.ops.lambertw import LambertW
from onnx_passes.ops.swish import Swish


# Inverse of the Swish function: As Swish is not invertible, this yields
# branches expressed in terms of the Lambert W function for inputs for which a
# real-valued solution of the inverse exists.
#
# All other inputs are mapped to the appropriate -/+ infinity, such that the
# inverse behaves nicely with respect to comparison (>=) and thus thresholding.
#
# Note: Only the real-valued primary (k=0) and secondary branch (k=-1) are
#  implemented. For any other value of k this yields NaN.
@register
def InverseSwish(x, alpha: float = 1.0, k: int = 0, tolerance: float = 1.0e-8):
    # Value at the global minimum of the Swish function is the smallest input
    # for which a real-valued solution of the inverse exists
    x_min = Swish(-alpha ** -1 * (1.0 + LambertW(op.Exp(-1.0))), alpha=alpha)

    # Short aliases to infinity and NaN used for out of range inputs
    inf = op.Constant(value_float=float("inf"))
    nan = op.Constant(value_float=float("nan"))

    # Evaluate the Lambert W function on the selected branch: For inputs with
    # real-valued solutions, the inverse is given as
    #   Swish^{-1}_{k}(x) = x + alpha**-1 * W_{k}(alpha * x * e**(-alpha * x))
    w = LambertW(alpha * x * op.Exp(-alpha * x), k, tolerance=tolerance)

    # Select from principal (k=0) and secondary (k=-1) branch or fall back
    # returning NaN for unsupported branches
    return op.Where(
        k == 0,
        op.Where(
            x >= x_min, x + alpha ** -1 * w, -inf
        ),
        op.Where(
            k == -1,
            op.Where(
                x < 0.0, op.Where(x >= x_min, x + alpha ** -1 * w, -inf), inf
            ),
            # Fallback: As ONNX Script needs to trace all branches and cannot
            # raise exceptions, NaN is returned for k not in {0,-1}
            nan
        )
    )


# As Silu is defined as the Swish functions with alpha=1.0, we can define the
# inverse of Silu as a special case of the inverse Swish.
@register
def InverseSilu(x, k: int = 0, tolerance: float = 1.0e-8):
    return InverseSwish(x, alpha=1.0, k=k, tolerance=tolerance)
