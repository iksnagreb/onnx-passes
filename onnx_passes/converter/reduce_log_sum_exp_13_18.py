# Collecting node attributes with optional defaults
from onnx_passes.passes.util import collect_attrs

from onnxscript.version_converter._version_converter import registry
from onnxscript import ir

register = registry.register


# Onnx 13 -> 18
@register("ReduceLogSumExp", node_version=13, up_conversion=True)
def convert_reduce_log_sum_exp_13_18(node: ir.Node, op):
    input = node.inputs[0]

    attributes = collect_attrs(
        node,
        {
            "keepdims": (ir.AttributeType.INT),
            "axes": (ir.AttributeType.INTS, ()),
        },
    )
    kwargs = {
                "keepdims": attributes["keepdims"].as_int(),
                "noop_with_empty_axes": 0,
                "axes": attributes["axes"].as_ints(),
            }
    
    return op.ReduceLogSumExp(input, **kwargs)
