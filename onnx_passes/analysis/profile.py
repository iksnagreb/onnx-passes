from pydantic import BaseModel, computed_field
from typing import Any


class Time(BaseModel, use_attribute_docstrings=True):
    """Profiling of start-end time differences."""

    start: float | int
    """Measured start time."""

    end: float | int
    """Measured end time."""

    @computed_field
    @property
    def difference(self) -> float | int:
        """Difference between start and end time."""
        return self.end - self.start


class Metrics(BaseModel, use_attribute_docstrings=True):
    """Profiling of named metrics of any type."""

    by_name: dict[str, Any] = {}
    """Metric value by name."""
