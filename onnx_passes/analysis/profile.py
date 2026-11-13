from pydantic import BaseModel, computed_field


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
