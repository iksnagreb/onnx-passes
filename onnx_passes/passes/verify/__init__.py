from onnx_passes.verify import Verify, metric

import onnx_ir as ir


@metric
class VerifyMetrics_v1(Verify):
    """Metric-based verification of the model without actually modifying."""

    @property
    def in_place(self) -> bool:
        return True

    @property
    def changes_input(self) -> bool:
        return True

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return ir.passes.PassResult(model, True)
