# ir.Value
import onnx_ir as ir

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# All streamlining passes are transformations derived from pattern-based rewrite
# rules
from onnx_passes.passes.base import Transformation, RewriteRulePass

# NumPy used for calculations on shapes and constant tensors in rewrites and
# match conditions
import numpy as np


# TODO: This could be generalized introducing more granularity, there could be
#  per-axis common scales for example
def _extract_common_scale(x: ir.Value) -> float:
    # Only constant tensors allow to pull out a common scale factor
    if (x := ir.convenience.get_const_tensor(x)) is not None:
        # Find the smallest absolute non-zero value in the tensor which will set
        # the scale (candidate scale, check for all other values being integer
        # multiples of this)
        scale = np.sort(np.unique(np.abs(x.numpy())))
        scale = scale[np.flatnonzero(scale)[0]]

        # Pulling out the scale and multiplying it back must yield the same
        # result within some tolerance
        if np.allclose(x.numpy(), scale * np.round(x.numpy() / scale)):
            return scale.item()
    # Common factor of 1.0 can trivially be pulled out of any tensor
    return 1.0


# Extracts common floating-point factors from constant inputs to MatMul if this
# results in a single scalar following the MatMul
@passes.verify.tolerance
@passes.register("factorize")
class ExtractCommonScaleFromMatMul(Transformation, RewriteRulePass):
    def pattern(self, op, x, y):
        return op.MatMul(x, y)

    def check(self, op, x, y):
        return _extract_common_scale(x) != 1 or _extract_common_scale(y) != 1

    def rewrite(self, op, x, y):
        # Common scales (or 1.0) of each of the two inputs
        scale_x = op.Constant(value_float=_extract_common_scale(x))
        scale_y = op.Constant(value_float=_extract_common_scale(y))

        # Round should not be injected on non factorized and especially not on
        # non-constant inputs
        if _extract_common_scale(x) != 1:
            x = op.Round(op.Div(x, scale_x))

        if _extract_common_scale(y) != 1:
            y = op.Round(op.Div(y, scale_y))

        # Replacement pattern: Divide common scale from inputs to MatMuls on the
        # input side and reintroduce both scales on the output side
        return op.Mul(op.MatMul(x, y), op.Mul(scale_x, scale_y))


# Extracts common floating-point factors from constant inputs to Gather if this
# results in a single scalar following the Gather. This only applies to the data
# input of the Gather operator.
@passes.verify.tolerance
@passes.register("factorize")
class ExtractCommonScaleFromGather(Transformation, RewriteRulePass):
    def pattern(self, op, data, indices):
        return op.Gather(data, indices)

    def check(self, op, data, indices):
        return _extract_common_scale(data) != 1

    def rewrite(self, op, data, indices):
        # Common scales (or 1.0) of the data input
        scale = op.Constant(value_float=_extract_common_scale(data))

        # Replacement pattern: Divide common scale from input to Gather on the
        # input side and reintroduce the same scale on the output side
        return op.Mul(op.Gather(op.Round(op.Div(data, scale)), indices), scale)
