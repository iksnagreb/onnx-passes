from onnx_passes.passes._base import Transformation
from onnx_passes.passes._verify import Verify

import onnx_ir as ir

# Common passes already implemented in ONNX IR
import onnx_ir.passes.common

_BLACKLIST: set[str] = {
    "Quant"
}


def _criterion(function: ir.Function) -> bool:
    """Decide whether to inline the function."""
    return function.name not in _BLACKLIST


class InlineFunctions_v1(Transformation, Verify):
    """Inline model local ONNX functions into the main graph."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return ir.passes.common.InlinePass(_criterion)(model)
