# ONNXScript IR subpackage (actually moved to its own package in more recent
# ONNX and ONNXScript versions)
from onnxscript import ir

# Topological sort pass built into ONNX IR and ONNXScript
from onnxscript.ir.passes.common import TopologicalSortPass

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import passes


# Performs topological sort on the entire model graph, reordering the nodes
@passes.register("cleanup")
@passes.register("topological-sort")
# class TopologicalSort(passes.base.Pass, ir.passes.InPlacePass):
class TopologicalSort(passes.base.Transformation):
    # Applies the built-in ONNX IR topological sort pass on a deep copy of the
    # model (as we prefer functional passes not modifying the original).
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return TopologicalSortPass()(ir.from_proto(ir.to_proto(model)))
