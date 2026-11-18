from onnx_passes.base import Analysis
from onnx_passes.eval import evaluate_model

from pathlib import Path

import onnx_ir as ir
import numpy as np

from onnx_passes.analysis.error import MaxAbsError


class EvaluateMaxAbsError_v1(Analysis):
    """Evaluate maximum absolute error of the model on reference data."""

    metadata_key = "max_abs_error"

    def analyze(self, model: ir.Model) -> MaxAbsError | None:
        if self.config.analysis.inputs and self.config.analysis.expected:
            inputs = []
            expected = []

            for file_or_value in self.config.analysis.inputs:
                if isinstance(file_or_value, Path | str):
                    file_or_value = np.load(file_or_value)
                inputs.append(file_or_value)

            for file_or_value in self.config.analysis.expected:
                if isinstance(file_or_value, Path | str):
                    file_or_value = np.load(file_or_value)
                expected.append(file_or_value)

            produced, _ = evaluate_model(model, inputs)

            return MaxAbsError.from_outputs(produced, expected)

        return None
