# ir.Model, ir.passes.PassResult, ir.from_proto, ir.to_proto, ...
import onnx_ir as ir

# Identity elimination pass build into ONNX IR and ONNXScript
from onnx_ir.passes.common import IdentityEliminationPass

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes


# Removes unnecessary Identity nodes from the graph
@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-identity")
class EliminateIdentity(passes.base.Transformation):
    # Applies the built-in ONNX IR Identity elimination pass on a deep copy of
    # the model (as we prefer functional passes not modifying the original).
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return IdentityEliminationPass()(ir.from_proto(ir.to_proto(model)))
