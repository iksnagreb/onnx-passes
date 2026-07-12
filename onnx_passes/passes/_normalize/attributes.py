from onnx_passes.passes._base import Transformation
from onnx_passes.passes._verify import Verify

import onnx_ir as ir

# Common passes already implemented in ONNX IR
import onnx_ir.passes.common


class AddDefaultAttributes_v1(Transformation, Verify):
    """Add default values for optional attributes not present in nodes."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return ir.passes.common.AddDefaultAttributesPass()(model)
