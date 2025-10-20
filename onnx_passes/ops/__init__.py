# Type annotation for callables
from typing import Callable

# ir.Model, ir.Value, ir.Attr, ir.tensor
import onnx_ir as ir

# Declaring a custom ONNX opset with name and version
from onnxscript.values import Opset
# Use onnxscript scripts for authoring custom operators as model local functions
from onnxscript import script, opset19 as op

# Custom operator domain for all operator defined in this package
domain, DOMAIN = Opset("onnx_passes.ops", 1), "onnx_passes.ops"

# Registry of custom operators defined with the onnx-passes package
_registry = {}


# Registers a custom operator defined via ONNX Script into the custom domain
# and injects this into any model for ONNX Runtime execution
def register(f: Callable):
    # Wrap the function as an OnnxFunction
    f = script(domain, default_opset=op)(f)
    # Add this function to the registry
    _registry[f.name] = f
    # Return the decorated function for chaining decorators
    return f


# Injects all registered custom operators into the model as model local
# functions
def inject_custom_ops(model: ir.Model):
    # Update the opset import to include the custom domain
    model.opset_imports[str(domain)] = domain.version
    # Convert from ONNX IR representation to proto representation which offers
    # access to the list of local functions
    model = ir.to_proto(model)
    # Add all registered function to the model
    model.functions.extend([f.to_function_proto() for f in _registry.values()])
    # Convert back to ONNX IR representation
    return ir.from_proto(model)


# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# Inserting custom ops is considered as an annotation pass as it does not really
# modify the model graph structure or values
from onnx_passes.passes.base import Annotation


# Annotation pass inserting custom operator functions into the model
@passes.register("inject-ops")
class InjectCustomOps(Annotation):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return ir.passes.PassResult(inject_custom_ops(model), False)
