# Path to files or directories
from pathlib import Path
# Define configuration structures as dataclasses
from dataclasses import dataclass, field
# Type hints for annotating dataclass members
from typing import Any


@dataclass(frozen=True)
class LoggingConfig:
    """Configuration options for logging pass application.

    Attributes
    ----------
    verbose : bool = False
        Print messages when entering/leaving/verifying/... passes
    checkpoint : bool | str = False
        Filename to use for checkpoints (or disable checkpointing)
    keep_intermediates : bool | Path = False
        Store intermediate models into this directory after each pass
    """

    verbose: bool = field(default=False)
    checkpoint: bool | str = field(default=False)
    keep_intermediates: bool | Path = field(default=False)


@dataclass(frozen=True)
class VerifyConfig:
    """Configuration options for pass verification.

    Attributes
    ----------
    tolerance : Tolerance = Tolerance(rtol=1.0e-5, atol=1.0e-8)
        Configuration options for tolerance-based pass verification
    metrics : tuple[Metric, ...] | None = None
        Configuration options for metrics-based pass verification
    full_context_dump : bool = False
        Save the full execution context including intermediate tensors
    inputs : list[Path | str | Any] = []
        Path to the verification reference inputs
    expected : list[Path | str | Any] = []
        Path to the verification reference outputs
    """

    @dataclass(frozen=True)
    class Tolerance:
        """Configuration options for tolerance-based pass verification.

        Attributes
        ----------
        rtol : float = 1.0e-5
            Relative verification tolerance
        atol : float = 1.0e-8
            Absolute verification tolerance
        """

        rtol: float = field(default=1.0e-5)
        atol: float = field(default=1.0e-8)

    @dataclass(frozen=True)
    class Metric:
        """Configuration options for metrics-based pass verification.

        Attributes
        ----------
        function : str
            Function evaluating the metric given produced and expected outputs
        range : tuple[float, float]
            Minimum and maximum metric value accepted for verification
        """

        function: str
        range: tuple[float, float]

    tolerance: Tolerance = field(default_factory=Tolerance)
    metrics: tuple[Metric, ...] | None = field(default=None)
    full_context_dump: bool = field(default=False)

    inputs: list[Path | str | Any] = field(default_factory=list)
    expected: list[Path | str | Any] = field(default_factory=list)


@dataclass(frozen=True)
class Config:
    """Top-level configuration for pass application.

    Attributes
    ----------
    logging : LoggingConfig
        Configuration options for logging pass application
    verify : VerifyConfig | bool
        Configuration options for pass verification
    """

    logging: LoggingConfig = field(default_factory=LoggingConfig)
    verify: VerifyConfig | bool = field(default_factory=VerifyConfig)
