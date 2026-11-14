from onnx_passes.base import Analysis

import onnx_ir as ir

from onnx_passes.analysis.multithreshold import MultiThresholdSummary


class SummarizeMultiThresholds_v1(Analysis):
    """Summarize MultiThresholds in the model broken down into categories."""

    metadata_key = "multithresholds"

    def analyze(self, model: ir.Model) -> MultiThresholdSummary:
        return MultiThresholdSummary.from_model(model)
