# ir.Attr and ir.AttributeType
import onnx_ir as ir

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# Derive Transformations (allowed to modify the graph) from pattern-based
# rewrite rules
from onnx_passes.passes.base import Transformation, RewriteRulePass

# Collecting node attributes with optional defaults
from onnx_passes.passes.util import collect_attrs


# Inlines LogSoftmax using Softmax.
@passes.verify.tolerance
@passes.register("inline-log-softmax")
class InlineLogSoftmax(Transformation, RewriteRulePass):
    def pattern(self, op, x):
        return op.LogSoftmax(x, _outputs=["y"])

    def rewrite(self, op, x, y):
        # Default according to ONNX operators reference documentation:
        #   https://onnx.ai/onnx/operators/onnx__LogSoftmax.html
        attributes = collect_attrs(
            y.producer(),
            {
                "axis": (ir.AttributeType.INT, -1),
            },
        )

        axis = attributes["axis"].as_int()
        return op.Log(op.Softmax(x, axis=axis))
