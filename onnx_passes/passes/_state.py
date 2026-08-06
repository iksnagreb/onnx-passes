# Define configuration structures as dataclasses
from dataclasses import dataclass, field
# Type hints for annotating dataclass members
from typing import Any


@dataclass(frozen=False)
class State:
    """Tracks state throughout pass applications

    Attributes
    ----------
    counter : int
        Global counter of pass applications
    history : list[type]
        History of applied passes by type
    verify : dict[str, VerifyState]
        History of verification states (execution context, metrics, etc.)
    """

    @dataclass(frozen=True)
    class VerifyState:
        """Verification state entry.

        Attributes
        ----------
        inputs : list[Any]
            Model inputs used for verification
        outputs : list[Any]
            Model outputs produced by verification
        expected : list[Any]
            Model outputs expected for verification
        context : dict[str, Any] | None = None
            Optional full execution context from verification
        metrics : dict[str, Any] | None = None
            Optional metrics from metric-based verification
        """

        inputs: list[Any]
        outputs: list[Any]
        expected: list[Any]

        context: dict[str, Any] | None = field(default=None)
        metrics: dict[str, Any] | None = field(default=None)

    counter: int = field(default=0)
    history: list[type] = field(default_factory=list)
    verify: dict[str, VerifyState] = field(default_factory=dict)

    @property
    def last(self):
        """The last pass applied."""
        return self.history[-1] if self.history else None

    @property
    def id(self):
        """Unique identifier of the last pass"""
        if self.last is not None:
            return f"{self.counter:08d}-{self.last.__name__}"  # noqa: Not None
        return None

    def log_pass(self, p):
        """Logs the pass p to the state dict, advancing counter and history."""
        self.counter = self.counter + 1
        self.history.append(type(p))

    def log_verification(self, inputs, outputs, expected, context, **metrics):
        """Logs a verification result to the state."""

        # Verification can only be done and logged after applying a pass - the
        # result will be associated with the last pass
        if self.last is None:
            raise RuntimeError(
                f"Tried to log verification result without ever logging a pass"
            )

        # Applied passes are uniquely identified by the class name and the
        # running counter
        self.verify[self.id] = State.VerifyState(  # noqa: Not None
            inputs, outputs, expected, context, metrics
        )
