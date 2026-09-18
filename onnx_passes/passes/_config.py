from pydantic import Field
from pydantic_settings import BaseSettings, CliToggleFlag

from pathlib import Path
from typing import Any, Optional, Callable


class LoggingConfig(BaseSettings, use_attribute_docstrings=True):
    """Configuration options for logging pass application."""

    verbose: CliToggleFlag[bool] = False
    """Print messages when entering/leaving/verifying/... each pass"""

    checkpoint: Optional[str] = None
    """Filename for checkpoints (or disable checkpointing)"""

    keep_intermediates: Optional[Path] = None
    """Directory to tore intermediate models after each pass"""


class VerifyConfig(BaseSettings, use_attribute_docstrings=True):
    """Configuration options for pass verification."""

    class Tolerance(BaseSettings, use_attribute_docstrings=True):
        """Configuration options for tolerance-based pass verification."""

        rtol: float = 1.0e-5
        """Relative verification tolerance"""

        atol: float = 1.0e-8
        """Absolute verification tolerance"""

    class Metric(BaseSettings, use_attribute_docstrings=True):
        """Configuration options for metrics-based pass verification."""

        function: str | Callable
        """Function evaluating the metric given produced and expected outputs"""

        range: tuple[float, float]
        """Minimum and maximum metric value accepted for verification"""

    tolerance: Tolerance = Tolerance()
    metrics: list[Metric] | None = None

    full_context_dump: CliToggleFlag[bool] = False
    """Save the full execution context including intermediate tensors"""

    inputs: list[Path | str | Any] = Field(default_factory=list)
    """Path to the verification reference inputs"""

    expected: list[Path | str | Any] = Field(default_factory=list)
    """Path to the verification reference outputs"""


class Config(BaseSettings, use_attribute_docstrings=True):
    """Top-level configuration for pass application."""

    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    """Configuration options for logging pass application"""

    verify: VerifyConfig | bool = Field(default_factory=VerifyConfig)
    """Configuration options for pass verification"""
