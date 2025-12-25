# Custom operator function registry
from onnx_passes.ops import register, op


@register
def ArgSort(x, axis: int = -1):
    # The following abuses the TopK operation to sort all elements along the
    # axis. To do so, we need to figure out the number of elements along axis.
    k = op.GatherElements(
        op.Shape(x),
        op.Expand(op.Constant(value_int=axis), op.Constant(value_ints=[1]))
    )

    # Abuse TopK to sort the input into ascending order (largest=0) along axis
    # by setting k = x.shape[axis]
    _, indices = op.TopK(x, k, axis=axis, largest=0)

    # Keep only the index tensor...
    return indices
