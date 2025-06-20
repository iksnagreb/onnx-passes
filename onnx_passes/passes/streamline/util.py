# ir.Value
import onnx_ir as ir
# np.all
import numpy as np


# Checks whether the ir.Value represents a constant: Either is_initializer or
# has a const_value set
def is_constant(v: ir.Value):
    return v.const_value is not None or v.is_initializer()


# Checks whether the two ir.Values are identical constants, i.e., all values are
# equal according to NumPy semantics
def identical_constants(a: ir.Value, b: ir.Value) -> bool:
    if is_constant(a) and is_constant(b):
        return np.all(a.const_value.numpy() == b.const_value.numpy())
    return False
