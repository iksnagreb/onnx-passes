# ir.Model, ir.passes.PassResult, ir.from_proto, ir.to_proto, ...
import onnx_ir as ir

# Unused node, function and opset removal passes build into ONNX IR
from onnx_ir.passes.common import (
    RemoveUnusedNodesPass, RemoveUnusedFunctionsPass, RemoveUnusedOpsetsPass
)

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes


# Removes unused nodes from the graph - wrapper around ONNX IR pass
@passes.verify.equality
@passes.register("cleanup")
@passes.register("remove-unused")
@passes.register("remove-unused-nodes")
class RemoveUnusedNodes(passes.base.Transformation):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return RemoveUnusedNodesPass()(model)


# Removes unused functions from the graph - wrapper around ONNX IR pass
@passes.verify.equality
@passes.register("remove-unused")
@passes.register("remove-unused-functions")
class RemoveUnusedFunctions(passes.base.Transformation):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return RemoveUnusedFunctionsPass()(model)


# Removes unused opset imports from the graph - wrapper around ONNX IR pass
@passes.verify.equality
@passes.register("remove-unused")
@passes.register("remove-unused-opsets")
class RemoveUnusedOpsets(passes.base.Transformation):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return RemoveUnusedOpsetsPass()(model)
