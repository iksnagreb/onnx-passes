from onnx_passes.passes._base import Pass

import onnx_ir as ir

# Common cleanup passes already implemented in ONNX IR, used here without any
# custom infrastructure.
import onnx_ir.passes.common


class Checker_v1(Pass):
    """Run the ONNX checker to check consistency of the model."""

    @property
    def in_place(self) -> bool:
        return True

    @property
    def changes_input(self) -> bool:
        return False

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return ir.passes.common.CheckerPass(full_check=True)(model)
