# ir.Model, ir.Value, ir.convenience.get_const_tensor
import onnx_ir as ir
# np.load for loading reference data, np.all
import numpy as np

# Base class for all custom ONNX IR passes developed in this library - this base
# class defines the (optional) interface for configuration and state tracking
from onnx_passes.passes.base import Pass


# Checks whether the ir.DataType is considered a signed data type: These are all
# signed integers as well as floating-point datatypes
# TODO: Replace all uses by data_type.is_signed() once available via ONNX IR,
#  see https://github.com/onnx/ir-py/pull/110
def is_signed(data_type: ir.DataType):
    return data_type in {
        ir.DataType.FLOAT,
        ir.DataType.INT8,
        ir.DataType.INT16,
        ir.DataType.INT32,
        ir.DataType.INT64,
        ir.DataType.FLOAT16,
        ir.DataType.DOUBLE,
        ir.DataType.COMPLEX64,
        ir.DataType.COMPLEX128,
        ir.DataType.BFLOAT16,
        ir.DataType.FLOAT8E4M3FN,
        ir.DataType.FLOAT8E4M3FNUZ,
        ir.DataType.FLOAT8E5M2,
        ir.DataType.FLOAT8E5M2FNUZ,
        ir.DataType.INT4,
        ir.DataType.FLOAT4E2M1,
    }


# Checks whether the ir.Value represents a constant: Either is_initializer or
# has a const_value set
def is_constant(v: ir.Value):
    return v.const_value is not None or v.is_initializer()


# Checks whether the ir.Value represents a scalar: Either the shape is empty or
# any dimension is of size 1
def is_scalar(v: ir.Value):
    return np.prod(v.shape) == 1


# Checks whether the two ir.Values are identical constants, i.e., all values are
# equal according to NumPy semantics
def identical_constants(a: ir.Value, b: ir.Value) -> bool:
    if is_constant(a) and is_constant(b):
        return bool(np.all(a.const_value.numpy() == b.const_value.numpy()))
    return False


# If v is a constant ir.Value (either from Constant op or initializer), returns
# the constant value as NumPy, otherwise returns None
def get_const_or_none(v: ir.Value):
    if (v := ir.convenience.get_const_tensor(v)) is not None:
        return v.numpy()
    return None


# Checks whether two potentially constant ir.Values match i.e., all values are
# equal according to NumPy semantics
def constant_match(a, b):
    if isinstance(a, ir.Value):
        a = get_const_or_none(a)
    if isinstance(b, ir.Value):
        b = get_const_or_none(b)
    return (a is not None or b is not None) and np.all(a == b)


# Injects pre- and post-condition methods into an ONNX IR pass, i.e., wraps and
# overwrites the .requires and .ensures methods.
def inject_pre_post_condition(cls: type[Pass], pre: callable, post: callable):
    # The wrapped pass might already have pre- and post-conditions defined which
    # we should preserve, adding the verification on top...
    _requires, _ensures = cls.requires, cls.ensures

    # Evaluate the new followed by the original pre-condition - we do this
    # afterward to preserve the order of operations when stacking decorators
    def requires(self: Pass, model: ir.Model) -> None:
        pre(self, model), _requires(self, model)

    # Evaluate the original followed by the new post-condition - we do this
    # first to preserve the order of operations when stacking decorators
    def ensures(self: Pass, model: ir.Model) -> None:
        _ensures(self, model), post(self, model)

    # Inject the new pre- and post-condition methods overwriting the exiting
    # methods which have been wrapped by the new ones.
    cls.requires, cls.ensures = requires, ensures
    # Return the modified class
    return cls


# Loads reference data from the config or state dictionary of an ONNX IR pass by
# first considering the state dictionary
def load_reference_data(p: Pass) -> tuple[list, list]:
    # Accessing non-existing dictionaries might result in AttributeError or
    # TypeError
    try:
        # First try the state dictionary if it contains a reference section
        if p.state_dict and "reference" in p.state_dict:
            return (p.state_dict["reference"].setdefault("inp", []),
                    p.state_dict["reference"].setdefault("out", []))

        # Make sure the next test does not result in KeyError or ValueError by
        # injecting empty default lists
        p.config["reference"].setdefault("inp", [])
        p.config["reference"].setdefault("out", [])

        # If no reference data is tracked via the state dictionary, this is
        # probably the first attempt at loading the data: Check the config
        if p.config and "reference" in p.config:
            return ([np.load(file) for file in p.config["reference"]["inp"]],
                    [np.load(file) for file in p.config["reference"]["out"]])

        # Nothing found, return two empty lists indicating no inputs/outputs
        return [], []
    # If the "references" section is not present, we might end up here
    except (AttributeError, TypeError):
        return [], []
