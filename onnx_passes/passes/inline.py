# ir.Model, ir.passes.PassResult, ir.from_proto, ir.to_proto, ...
import onnx_ir as ir

# Function inlining pass build into ONNX IR and ONNXScript
from onnx_ir.passes.common import InlinePass

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes


# Performs function inlining on the entire model graph
@passes.verify.equality
@passes.register("inline")
@passes.register("inline-functions")
class InlineFunctions(passes.base.Transformation):
    # Applies the built-in ONNX IR function inline pass on a deep copy of the
    # model (as we prefer functional passes not modifying the original).
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return InlinePass()(ir.from_proto(ir.to_proto(model)))
