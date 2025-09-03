# ir.Model, ir.passes.PassResult, ir.from_proto, ir.to_proto, ...
import onnx_ir as ir

# Common subexpression elimination pass build into ONNX IR and ONNXScript
from onnx_ir.passes.common import CommonSubexpressionEliminationPass

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes


# Removes common subexpression from the graph, e.g., moving the same operator
# upwards as forks
@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-common-subexpression")
class EliminateCommonSubexpression(passes.base.Transformation):
    # Applies the built-in ONNX IR common subexpression elimination pass on a
    # deep copy of the model (as we prefer functional passes not modifying the
    # original).
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return CommonSubexpressionEliminationPass()(
            ir.from_proto(ir.to_proto(model))
        )
