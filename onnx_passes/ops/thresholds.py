# Datatype annotations ir.DataType.FLOAT
import onnx_ir as ir

# Custom operator function registry
from onnx_passes.ops import register, op


@register
def MultiThreshold(x, thresholds, weights):  # noqa: Operator name is uppercase
    # Comparison of inputs and all corresponding thresholds: Expand input
    # dimensions to match the threshold parameter shape via broadcasting
    steps = op.GreaterOrEqual(op.Unsqueeze(x, axes=[-1]), thresholds)
    # Type-casing turns boolean unit steps to reducible floats
    steps = op.Cast(steps, to=ir.DataType.FLOAT)
    # Finally the multi-threshold output reduces over all steps removing the
    # previously expanded dimension
    return op.ReduceSum(weights * steps, [-1], keepdims=0)
