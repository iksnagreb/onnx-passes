# ir.Model, ir.passes.PassResult, ...
import onnx_ir as ir

# Model checker pass build into ONNX IR and ONNXScript
from onnx_ir.passes.common import CheckerPass

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes


# Runs the ONNX model checker on the entire model graph - checks for consistency
# of IR/Opset versions and domains, raises an exception if something is wrong
@passes.register("checker")
class Checker(passes.base.Analysis):
    # Applies the built-in ONNX IR model checker pass on the model without
    # modifying anything.
    #
    # Configuration options can be supplied via the "model_checker" field of
    # the configuration dictionary referenced by the pass base.
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return CheckerPass(**self.config.setdefault("model_checker", {}))(model)
