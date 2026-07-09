# Build a class hierarchy of ONNX passes starting from ABC
from abc import ABC
# Path to files or directories
from pathlib import Path
# Define configuration structures as dataclasses
from dataclasses import asdict, astuple
# Type hints for annotating dataclass members
from typing import Any
# Verification styles are selected via enums
from enum import Enum

# ONNX intermediate representation
import onnx_ir as ir
# Numpy arrays are used for verification inputs/outputs
import numpy as np

# Use the ONNX reference evaluator to evaluate the model for verification
from onnx.reference import ReferenceEvaluator

# Pass configuration and state tracking structures and the base class for all
# passes
from onnx_passes.passes._config import Config
from onnx_passes.passes._state import State
from onnx_passes.passes._base import Pass, VerificationError


def _evaluate_model(model: ir.Model, inputs: list[Any]):
    """Evaluates the model using the ONNX reference evaluator."""

    # Fill the execution context with the input values assuming they
    # correspond to the model graph inputs in correct order
    context = {}

    for tensor, value in zip(model.graph.inputs, inputs):
        if (dtype := tensor.dtype) is not None:
            value = np.asarray(value, dtype.numpy())
        context[tensor.name] = np.asarray(value)

    # Use the ONNX reference evaluator to execute the model on the context
    # collecting named outputs into the context as well
    session = ReferenceEvaluator(ir.to_proto(model))
    context = session.run(None, context, intermediate=True)

    # Collect the outputs from the execution context assuming they
    # correspond to the model graph outputs in correct order
    outputs = []

    for tensor in model.graph.outputs:
        outputs.append(np.asarray(context[tensor.name]))  # noqa: Not None...?

    return outputs, context


# Use importlib for dynamic imports
from importlib import import_module


def _resolve_metric(identifier: str):
    """Resolves the metric evaluation function from the identifier."""

    # First try to look up the function in the global scope, if it is not there,
    # try interpreting the identifier as a fully qualified name
    try:
        return globals()[identifier]
    except KeyError:
        try:
            *module, function = identifier.split(".")
            return getattr(import_module(".".join(module)), function)
        except ValueError or KeyError or ModuleNotFoundError:
            raise KeyError(f"Could not resolve metric: {identifier}")


class Verify(Pass, ABC):
    """Makes a pass verify the model after application."""

    class Method(Enum):
        """Selectors for verification methods."""
        EQUALITY = 0
        TOLERANCE = 1
        METRIC = 2

    _method: Method = Method.EQUALITY

    _inputs: list[Any]
    _outputs: list[Any]
    _expected: list[Any]

    def __init__(self, config: Config = Config()):
        """Initializes the pass with a pass configuration."""
        super().__init__(config)

        # Clear the verification data
        self._inputs = []
        self._outputs = []
        self._expected = []

    def prepare_verify(self, model: ir.Model):
        """Prepare model verification before applying the pass."""

        # Skip verification if disabled globally by setting the configuration
        # entry to False
        if not self.config.verify:
            if self.config.logging.verbose:
                print(
                    f"Skipping verification of {self.identifier}:"
                    f" Verification not enabled"
                )
            return

        # Load the reference inputs and expected outputs from files if not
        # already loaded
        if not self._inputs:
            for file_or_value in self.config.verify.inputs:  # noqa: Not bool
                if isinstance(file_or_value, Path | str):
                    file_or_value = np.load(file_or_value)
                self._inputs.append(file_or_value)

        if not self._expected:
            for file_or_value in self.config.verify.expected:  # noqa: Not bool
                if isinstance(file_or_value, Path | str):
                    file_or_value = np.load(file_or_value)
                self._expected.append(file_or_value)

        # Equality- and tolerance-based verification compares before and after
        # results instead of some fixed expectation
        if self._method in {Verify.Method.EQUALITY, Verify.Method.TOLERANCE}:
            # Use the ONNX reference evaluator to execute the model on the
            # inputs collecting expected outputs
            try:
                self._expected, _ = _evaluate_model(model, self._inputs)
            except RuntimeError as e:
                raise VerificationError(
                    f"Verification of pass '{self.identifier}' failed"
                ) from e

    def verify(self, result: ir.passes.PassResult):
        """Verify the model after applying the pass."""

        # Skip verification if disabled globally by setting the configuration
        # entry to False or if the model has not been changed by the pass
        if not self.config.verify:
            if self.config.logging.verbose:
                print(
                    f"Skipping verification of {self.identifier}:"
                    f" Verification not enabled"
                )
            return

        if not result.modified:
            if self.config.logging.verbose:
                print(
                    f"Skipping verification of {self.identifier}:"
                    f" Model not modified"
                )
            return

        # Use the ONNX reference evaluator to execute the model on the inputs
        # collecting outputs and the full execution context as well
        try:
            self._outputs, context = _evaluate_model(result.model, self._inputs)
        except RuntimeError as e:
            raise VerificationError(
                f"Verification of pass '{self.identifier}' failed"
            ) from e

        # Discard the full execution context if not requested for debugging
        # purposes
        if not self.config.verify.full_context_dump:  # noqa: Not bool
            context = None

        # Compare the produced to the expected outputs depending on the selected
        # verification method
        metrics = {}

        if self._method == Verify.Method.METRIC:
            for metric in self.config.verify.metrics:  # noqa: Not None
                key, _ = astuple(metric)
                function = _resolve_metric(key)
                metrics[key] = function(self._outputs, self._expected)

        # Log the verification inputs and outputs to the metadata for debugging
        # before evaualting the condition and potentially raising expections
        result.model.meta.setdefault("passes", State()).log_verification(
            self._inputs, self._outputs, self._expected, context, **metrics
        )

        # Compare for *strict* equality of *all* values from *all* outputs
        if self._method == Verify.Method.EQUALITY:
            for tensor, x, y in zip(
                    result.model.graph.outputs, self._outputs, self._expected
            ):
                if np.any(x != y):
                    raise VerificationError(
                        f"Output '{tensor.name}' not as expected"
                    )

        # Compare equality within tolerance of *all* values from *all* outputs
        if self._method == Verify.Method.TOLERANCE:
            for tensor, x, y in zip(
                    result.model.graph.outputs, self._outputs, self._expected
            ):
                if not np.allclose(
                        x, y, **asdict(self.config.verify.tolerance)  # noqa
                ):
                    raise VerificationError(
                        f"Output {tensor.name} not within tolerance"
                    )

        # Check whether all metrics lie within the required range and raise
        # exception if not
        if self._method == Verify.Method.METRIC:
            for metric in self.config.verify.metrics:  # noqa: Not None
                key, (_min, _max) = astuple(metric)
                if not _min <= (value := metrics[key]) <= _max:
                    raise VerificationError(
                        f"{key} {value} not within [{_min}, {_max}] as required"
                    )

        if self.config.logging.verbose:
            print(
                f"Successfully verified model after {self.identifier}"
            )


def equality(cls):
    """Marks a pass to verify the model for strict equality."""

    if not issubclass(cls, Verify):
        raise SyntaxError(f"Decorator only applies to verified passes: {cls}")

    cls._method = Verify.Method.EQUALITY

    return cls


def tolerance(cls):
    """Marks a pass to verify the model within tolerance."""

    if not issubclass(cls, Verify):
        raise SyntaxError(f"Decorator only applies to verified passes: {cls}")

    cls._method = Verify.Method.TOLERANCE

    return cls


def metric(cls):
    """Marks a pass to verify the model via metrics."""

    if not issubclass(cls, Verify):
        raise SyntaxError(f"Decorator only applies to verified passes: {cls}")

    cls._method = Verify.Method.METRIC

    return cls


@metric
class VerifyMetrics_v1(Verify):
    """Metric-based verification of the model without actually modifying."""

    @property
    def in_place(self) -> bool:
        return True

    @property
    def changes_input(self) -> bool:
        return True

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return ir.passes.PassResult(model, True)
