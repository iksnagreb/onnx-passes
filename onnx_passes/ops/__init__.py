# Inspecting python objects: Used to get function signature for extracting type
# annotations
import inspect

# Implementing and registering custom operators with the ai.onnx.contrib domain
from onnxruntime_extensions import onnx_op, PyOp, default_opset_domain

# Create some readable aliases to the PyOp type definitions resembling the ONNX
# Script annotations
# TODO: Extend this list or find some other mechanism...
FLOAT = PyOp.dt_float
INT32 = PyOp.dt_int32
INT64 = PyOp.dt_int64
STRING = PyOp.dt_string
BOOL = PyOp.dt_bool

# The domain of custom operators registered with onnxruntime_extensions PyOp
DOMAIN = default_opset_domain()


# Registers a pyton function as implementation of a custom ONNX operator op_type
def register_op(op_type: str | None = None, attrs: set | list | tuple = ()):
    # Assume FLOAT types if no type annotation is present
    def default_float(annotation: inspect.Parameter):
        return annotation if annotation != inspect.Parameter.empty else FLOAT

    # Inner decorator actually registering the wrapped function
    def inner(f: callable):
        # Inspect the function signature for input and output type annotations
        signature = inspect.signature(f)
        # Collect all function parameters with optional annotations
        inputs = signature.parameters

        # First extract attribute annotations from the explicitly specified list
        # of named attributes
        attributes = {attr: inputs[attr].annotation for attr in attrs}

        # Collect all inputs which are not already covered by named attributes
        inputs = inputs.values()
        inputs = [param for param in inputs if param.name not in attrs]
        # Turn missing input type annotations to FLOAT by default
        inputs = [default_float(param.annotation) for param in inputs]

        # Outputs must always be wrapped as lists or tuples
        if not isinstance(outputs := signature.return_annotation, tuple):
            outputs = [outputs]
        # Turn missing output type annotations to FLOAT by default
        outputs = [default_float(annotation) for annotation in outputs]

        # Default to the function name if no op_type is given
        name = op_type if op_type is not None else f.__name__

        # Instantiate the ONNX Runtime Extension decorator for registering the
        # wrapped function
        return onnx_op(
            op_type=name, inputs=inputs, outputs=outputs, attrs=attributes
        )(f)

    # Inner decorator wrapping the custom operator function
    return inner
