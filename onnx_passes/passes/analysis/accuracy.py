from onnx_passes.base import Analysis
from onnx_passes.eval import evaluate_model

from pathlib import Path

import onnx_ir as ir
import numpy as np

from onnx_passes.analysis.accuracy import Accuracy


class EvaluateAccuracy_v1(Analysis):
    """Evaluate classification accuracy of the model on reference data."""

    metadata_key = "accuracy"

    def analyze(self, model: ir.Model) -> Accuracy | None:
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

            if len(expected) > 1 or len(produced) > 1:
                if self.config.logging.verbose:
                    print(
                        f"{self.identifier}: Model produces multiple outputs"
                        f" - using first to calculate accuracy"
                    )

            expected = expected[0]
            produced = produced[0]

            return Accuracy.from_outputs(produced, expected)

        return None
