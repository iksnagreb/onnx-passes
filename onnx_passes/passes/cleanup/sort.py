# ir.Model, ir.passes.PassResult, ir.from_proto, ir.to_proto, ...
import onnx_ir as ir

# Topological sort pass build into ONNX IR and ONNXScript
from onnx_ir.passes.common import TopologicalSortPass

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes


# Performs topological sort on the entire model graph, reordering the nodes
@passes.verify.equality
@passes.register("cleanup")
@passes.register("topological-sort")
class TopologicalSort(passes.base.Transformation):
    # Applies the built-in ONNX IR topological sort pass on a deep copy of the
    # model (as we prefer functional passes not modifying the original).
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return TopologicalSortPass()(ir.from_proto(ir.to_proto(model)))
