# ir.Model, ir.passes.PassResult, ir.from_proto, ir.to_proto, ...
import onnx_ir as ir

# Common passes from ONNX IR for cleaning up constants: Prefer all constants to
# be initializers, but do not track these as graph inputs
from onnx_ir.passes.common import (
    LiftConstantsToInitializersPass,
    LiftSubgraphInitializersToMainGraphPass,
    RemoveInitializersFromInputsPass,
    DeduplicateInitializersPass
)

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes


# Remove initializers from inputs - wrapper around ONNX IR pass
#
# Note: As this enables different ONNX Runtime constant optimizations, this
# pass does not preserve equality, only equality within tolerance.
@passes.verify.tolerance
@passes.register("cleanup")
@passes.register("remove-initializers-from-inputs")
class RemoveInitializersFromInputs(passes.base.Transformation):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return RemoveInitializersFromInputsPass()(model)


# Lift constants to initializers - wrapper around ONNX IR pass
@passes.verify.equality
@passes.register("cleanup")
@passes.register("lift-constants")
class LiftConstantsToInitializers(passes.base.Transformation):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Load optional configuration parameters - defaults to what is specified
        # by the ONNX IR
        config = self.config.setdefault("lift_constants", {
            "lift_all_constants": True, "size_limit": 0
        })
        # Apply the built-in ONNX IR initializer lift pass
        return LiftConstantsToInitializersPass(**config)(model)


# Lift subgraph initializers to main graph - wrapper around ONNX IR pass
@passes.verify.equality
@passes.register("cleanup")
@passes.register("lift-constants")
class LiftSubgraphInitializersToMainGraph(passes.base.Transformation):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return LiftSubgraphInitializersToMainGraphPass()(model)


# Deduplicates initializers - wrapper around ONNX IR pass
@passes.verify.equality
@passes.register("cleanup")
@passes.register("deduplicate-initializers")
class DeduplicateInitializers(passes.base.Transformation):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Load optional configuration parameters - defaults to what is specified
        # by the ONNX IR
        config = self.config.setdefault("deduplicate_initializers", {})
        # Apply the built-in ONNX IR initializer deduplication pass
        return DeduplicateInitializersPass(**config)(model)
