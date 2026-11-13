from pydantic import BaseModel, computed_field

import numpy as np


class Accuracy(BaseModel, use_attribute_docstrings=True):
    """Model accuracy analysis results."""

    predictions: int
    """Total number of predictions."""

    correct_top_1: int
    """Number of correct predictions in the top-1 predictions."""

    correct_top_5: int
    """Number of correct predictions in the top-5 predictions."""

    @computed_field
    @property
    def top_1(self) -> float:
        """Top-1 accuracy: fraction of correct predictions within the top-1."""
        return self.correct_top_1 / self.predictions

    @computed_field
    @property
    def top_5(self) -> float:
        """Top-5 accuracy: fraction of correct predictions within the top-5."""
        return self.correct_top_5 / self.predictions

    @classmethod
    def from_outputs(cls, produced, expected) -> "Accuracy":
        """Calculate accuracy from produced and expected outputs."""

        # If the shape of the expected classes indicates these are class
        # probabilities as well, summarize these first.
        if expected.shape == produced.shape:
            expected = expected.argmax(axis=-1, keepdims=True)

        # Collect top-1 and top-5 predictions along the final axis, i.e., for
        # spatial or sequential outputs there are predictions per pixel/step.
        top_1 = np.sum(
            (produced.argsort(axis=-1)[..., -1:] == expected).any(axis=-1)
        )

        top_5 = np.sum(
            (produced.argsort(axis=-1)[..., -5:] == expected).any(axis=-1)
        )

        predictions = expected.size

        return Accuracy(
            predictions=predictions, correct_top_1=top_1, correct_top_5=top_5
        )
