from onnx_passes.passes._base import Transformation
from onnx_passes.passes._verify import Verify

import onnx_ir as ir


class Cleanup_v1(Transformation, Verify):
    """Basic cleanup of the model without transforming anything."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return ir.passes.PassResult(model, True)
