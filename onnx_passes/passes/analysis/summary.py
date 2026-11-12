from onnx_passes.base import Analysis

import onnx_ir as ir

from onnx_passes.analysis.summary import OperationCounts
from onnx_passes.analysis.summary import ParameterCounts
from onnx_passes.analysis.summary import AttributeCounts
from onnx_passes.analysis.summary import ModelTopology


class CountOperations_v1(Analysis):
    """Count operations in the model broken down into categories."""

    metadata_key = "operations"

    def analyze(self, model: ir.Model) -> OperationCounts:
        return OperationCounts.from_model(model)


class CountParameters_v1(Analysis):
    """Count parameters in the model broken down into categories."""

    metadata_key = "parameters"

    def analyze(self, model: ir.Model) -> ParameterCounts:
        return ParameterCounts.from_model(model)


class CountAttributes_v1(Analysis):
    """Count attributes in the model broken down into categories."""

    metadata_key = "attributes"

    def analyze(self, model: ir.Model) -> AttributeCounts:
        return AttributeCounts.from_model(model)


class SummarizeTopology_v1(Analysis):
    """Summarize the model topology."""

    metadata_key = "topology"

    def analyze(self, model: ir.Model) -> ModelTopology:
        return ModelTopology.from_model(model)
