# Custom operator function registry
from onnx_passes.ops import register, op


# In pure ONNX lower the convolution input generator Im2Col as
# gathering from the flattened input at precomputed indices
@register
def Im2Col(x, indices):  # noqa: Operator name is uppercase
    return op.Gather(op.Flatten(x), indices, axis=1)
