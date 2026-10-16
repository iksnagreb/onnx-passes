from onnx_passes.passes.base import Transformation, _cleanup
from onnx_passes.passes.verify import Verify

import onnx_ir as ir


class Cleanup_v1(Transformation, Verify):
    """Basic cleanup of the model without transforming anything."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return _cleanup(model)
