# Collecting node attributes with optional defaults
from collections.abc import Sequence

from onnx_passes.passes.util import collect_attrs

from onnxscript.version_converter._version_converter import registry
from onnxscript import ir
import onnxscript.ir._tape as _tape

RewriterContext = _tape.Builder
ReturnValue = Sequence[ir.Value] | ir.Value | None

register = registry.register


# Onnx 13 -> 18
@register("ReduceLogSumExp", node_version=18, up_conversion=True)
def convert_reduce_log_sum_exp_13_18(node: ir.Node, op: RewriterContext) -> ReturnValue:
    inp = node.inputs[0]

    attributes = collect_attrs(
        node,
        {
            "keepdims": (ir.AttributeType.INT, 1),
            "axes": (ir.AttributeType.INTS, ()),
        },
    )
    kwargs = {
        "keepdims": attributes["keepdims"].as_int(),
        "noop_with_empty_axes": 0,
    }

    return op.ReduceLogSumExp(inp, op.Constant(value_ints=attributes["axes"].as_ints()), **kwargs)
