# Custom operator function registry
from onnx_passes.ops import register, op


# Custom rounding function with configurable rounding mode via attribute
# matching QONNX/Brevitas behavior. Primarily meant to be used inside the Quant
# operator implementation.
@register
def Round(x, rounding_mode: str = "ROUND"):  # noqa: Operator name is uppercase
    # Predefined rounding mode constants for readability...
    ROUND = op.Constant(value_string="ROUND")
    CEIL = op.Constant(value_string="CEIL")
    FLOOR = op.Constant(value_string="FLOOR")
    ROUND_TO_ZERO = op.Constant(value_string="ROUND_TO_ZERO")

    # It is not possible to return from within If-Else branches, assign to a
    # temporary within each branch
    if rounding_mode == ROUND:
        y = op.Round(x)
    elif rounding_mode == CEIL:
        y = op.Ceil(x)
    elif rounding_mode == FLOOR:
        y = op.Floor(x)
    elif rounding_mode == ROUND_TO_ZERO:
        y = op.Mul(op.Sign(x), op.Floor(op.Abs(x)))
    else:
        # Else branch cannot be omitted, and it is not possible to raise
        # exceptions or assertions - fallback to Round...
        y = op.Round(x)

    # Return output from branch selected by rounding mode
    return y
