from onnx_passes.passes._base import RewriteRule
from onnx_passes.passes._verify import Verify

from onnx_passes.traits.elementwise import produced_by_elementwise

from itertools import dropwhile

import onnx_ir as ir
import numpy as np


def unbroadcast(x: np.ndarray, squeeze: bool = True) -> np.ndarray:
    """Unbroadcast redundant dimensions from a NumPy array."""

    for axis in range(x.ndim):
        y = x.swapaxes(0, axis)

        if np.all(y[:1] == y):
            x = y[:1].swapaxes(0, axis)

    if squeeze:
        x = np.reshape(x, (*dropwhile(lambda size: size == 1, x.shape),))

    return x


class UnbroadcastElementwise_v1(RewriteRule, Verify):
    """Remove redundant dimensions from constant elementwise inputs."""

    @staticmethod
    def pattern(op):
        return op.submodule("")(_allow_other_inputs=True, _outputs=["out"])

    @staticmethod
    def check(context, out):
        if produced_by_elementwise(context, out):
            for x in out.producer().inputs:
                if (x := ir.convenience.get_const_tensor(x)) is not None:
                    if unbroadcast(x.numpy()).shape != x.numpy().shape:
                        return out.shape is not None and out.shape.is_static()
        return False

    @staticmethod
    def rewrite(op, out):
        # Find the elementwise operator which produces the matched value (the
        # value level check guarantees this exists and is indeed the node we are
        # interested in).
        elementwise = out.producer()

        # Collect the list of inputs to the elementwise operation with all
        # constant inputs unbroadcast if possible.
        inputs = []

        for inp in elementwise.inputs:

            if (x := ir.convenience.get_const_tensor(inp)) is not None:
                inp = op.Constant(value=ir.tensor(unbroadcast(x.numpy())))

            inputs.append(inp)

        # Insert the replacement pattern with attributes transplanted from the
        # elementwise operator and final output expanded
        return op.Expand(
            op.op(
                elementwise.op_type, *inputs, **elementwise.attributes
            ),
            op.Constant(value_ints=out.shape[:])
        )
