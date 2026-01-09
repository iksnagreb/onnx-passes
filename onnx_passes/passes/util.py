# ir.Model, ir.Value, ir.convenience.get_const_tensor
import onnx_ir as ir

# np.load for loading reference data, np.all
import numpy as np

# Used to remove leading dimensions from array/tensor shapes by condition
from itertools import dropwhile

# Base class for all custom ONNX IR passes developed in this library - this base
# class defines the (optional) interface for configuration and state tracking
from onnx_passes.passes.base import Pass


# Collects attributes of a Node into a dictionary inserting defaults if the
# attribute is not present
def collect_attrs(node: ir.Node, attrs: dict):
    attrs = attrs.copy()
    for key, (_type, default) in attrs.items():
        if not (attr := node.attributes.get(key, None)):
            attr = ir.Attr(key, _type, default)
        attrs[key] = attr
    return attrs


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
    return v.shape is not None and v.shape.is_static() and np.prod(v.shape) == 1


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
    def ensures(self: Pass, result: ir.passes.PassResult) -> None:
        _ensures(self, result), post(self, result)

    # Inject the new pre- and post-condition methods overwriting the exiting
    # methods which have been wrapped by the new ones.
    cls.requires, cls.ensures = requires, ensures
    # Return the modified class
    return cls


# Loads reference data from the config or state dictionary of an ONNX IR pass by
# first considering the state dictionary
def load_reference_data(p: Pass) -> tuple[list, list]:
    # List of verification input and output files, defaults to empty lists
    inp = p.config.setdefault("reference", {}).setdefault("inp", [])
    out = p.config.setdefault("reference", {}).setdefault("out", [])

    def _load(file_or_array):
        if isinstance(file_or_array, np.ndarray):
            return file_or_array
        return np.load(file_or_array)

    # Load each file into a NumPy array and return two lists of inputs and
    # expected outputs
    return [_load(file) for file in inp], [_load(file) for file in out]


# Expands a constant of True to the shape of the input x
def true_like(op, x):
    return op.Expand(
        op.Cast(op.Constant(value_int=1), to=ir.DataType.BOOL), op.Shape(x)
    )


# Expands a constant of 1 to the shape of the input x
def ones_like(op, x):
    return op.Expand(op.CastLike(op.Constant(value_int=1), x), op.Shape(x))


# Expands a constant of False to the shape of the input x
def false_like(op, x):
    return op.Expand(
        op.Cast(op.Constant(value_int=0), to=ir.DataType.BOOL), op.Shape(x)
    )


# Expands a constant of 0 to the shape of the input x
def zeros_like(op, x):
    return op.Expand(op.CastLike(op.Constant(value_int=0), x), op.Shape(x))


# Reverse broadcasting of an array according to NumPy broadcasting semantics and
# also squeezes all leading dimensions of size 1.
def unbroadcast(x, squeeze=True, approximate=False, rtol=1.0e-5, atol=1.0e-4):
    # Start collecting a list of slices which can be used to index into the
    # array to remove repeated dimensions
    slices = []

    # Check each dimension if it is repeated, i.e., taking the first entry can
    # be expanded back to match the whole array
    for i in range(x.ndim):
        if np.all(x[(*slices, slice(0, 1))] == x):
            # Select [0,1) from the array, un-broadcasting dimension i
            slices.append(slice(0, 1))
        elif (approximate
              and np.allclose(x[(*slices, slice(0, 1))], x, rtol, atol)):
            slices.append(slice(0, 1))
        else:
            # Select all elements from the array
            slices.append(slice(0, None))

    # Apply un-broadcasting but keep all axes of size 1 for now
    y = x[(*slices,)]

    # Squeeze leading dimensions of size 1 as these can be restored when using
    # the array in broadcasting expressions
    if squeeze:
        return np.reshape(y, (*dropwhile(lambda size: size == 1, y.shape),))

    # Explicitly leave unbroadcast dimensions marked as 1, i.e., keep the rank
    # but remove redundancies
    return y
