# ir.Model, ir.passes.PassResult, ir.from_proto, ir.to_proto, ...
import onnx_ir as ir

# Shape inference pass build into ONNX IR and ONNXScript
from onnx_ir.passes.common import ShapeInferencePass

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes


# Performs shape inference on the entire model graph, adding or updating the
# shape annotations wherever possible
@passes.verify.equality
@passes.register("shape-inference")
class ShapeInference(passes.base.Annotation):
    # Applies the built-in ONNX IR shape inference pass on a deep copy of the
    # model (as we prefer functional passes not modifying the original).
    #
    # Configuration options can be supplied via the "shape_inference" field of
    # the configuration dictionary referenced by the transformation base.
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Load optional configuration parameters - defaults to what is specified
        # by the ONNX IR
        config = self.config.setdefault("shape_inference", {})
        # Apply the built-in ONNX IR shape inference pass on a deep copy of the
        # model
        return ShapeInferencePass(**config)(ir.from_proto(ir.to_proto(model)))
