from onnx_passes.passes._base import Transformation
from onnx_passes.passes._verify import Verify

import onnx_ir as ir

# Common cleanup passes already implemented in ONNX IR, used here without any
# custom infrastructure.
import onnx_ir.passes.common

_SIZE_LIMIT: int = 2 ** 32


class CommonSubexpressionElimination_v1(Transformation, Verify):
    """Eliminates common subexpressions from the model."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return ir.passes.common.CommonSubexpressionEliminationPass(_SIZE_LIMIT)(
            model
        )
