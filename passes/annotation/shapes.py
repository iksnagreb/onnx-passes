# Deep copies to have functional transformations, i.e., not modifying the
# original
import copy

# ONNXScript IR subpackage (actually moved to its own package in more recent
# ONNX and ONNXScript versions)
from onnxscript import ir

# Shape inference pass built into ONNX IR and ONNXScript
from onnxscript.ir.passes.common import ShapeInferencePass

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import passes


# Performs shape inference on the entire model graph, adding or updating the
# shape annotations wherever possible
@passes.register("annotation")
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
        return ShapeInferencePass(**config)(copy.deepcopy(model))
