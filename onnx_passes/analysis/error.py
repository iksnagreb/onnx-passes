from pydantic import BaseModel, computed_field

import numpy as np


class MaxAbsError(BaseModel, use_attribute_docstrings=True):
    """Model maximum absolute error analysis result."""

    per_output: list[float] = []
    """Maximum absolute error per model output."""

    @computed_field()
    @property
    def total(self) -> float:
        """Maximum absolute error over all model outputs."""
        return max(self.per_output)

    @classmethod
    def from_outputs(cls, produced, expected) -> "MaxAbsError":
        """Calculate maximum absolute error from produced and expected outputs.
        """

        max_abs_error = MaxAbsError()

        for i, (x, y) in enumerate(zip(produced, expected)):
            try:
                max_abs_error.per_output.append(float(np.max(np.abs(x - y))))
            except TypeError:
                # Try fallback interpreting all values as 32-bit floats
                x = x.astype(np.float32)
                y = y.astype(np.float32)

                max_abs_error.per_output.append(float(np.max(np.abs(x - y))))

        return max_abs_error
