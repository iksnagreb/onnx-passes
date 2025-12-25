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
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return TopologicalSortPass()(model)
