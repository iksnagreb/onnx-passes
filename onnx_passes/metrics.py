from onnx_passes.analysis.accuracy import Accuracy
from onnx_passes.analysis.error import MaxAbsError

from typing import Callable
from importlib import import_module


def resolve_metric(identifier: str | Callable):
    """Resolves the metric evaluation function from the identifier."""

    # Pass through already resolved callables
    if isinstance(identifier, Callable):
        return identifier

    identifier: str = str(identifier)

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


def max_abs_error(produced: list, expected: list) -> float:
    """Calculate the maximum absolute error between each pair of elements."""
    return MaxAbsError.from_outputs(produced, expected).total


def accuracy_top_1(produced: list, expected: list) -> float:
    """Calculate top-1 classification accuracy for produced probabilities.

    Assumes the model produces a single output tensor with class probabilities
    or labels along its final axis. Preceding axes will be flattened.
    """
    return Accuracy.from_outputs(produced[0], expected[0]).top_1


def accuracy_top_5(produced: list, expected: list) -> float:
    """Calculate top-5 classification accuracy for produced probabilities.

    Assumes the model produces a single output tensor with class probabilities
    or labels along its final axis. Preceding axes will be flattened.
    """
    return Accuracy.from_outputs(produced[0], expected[0]).top_5


def accuracy(produced: list, expected: list) -> float:
    """Calculate top-1 classification accuracy for produced probabilities.

    Assumes the model produces a single output tensor with class probabilities
    or labels along its final axis. Preceding axes will be flattened.
    """
    return accuracy_top_1(produced, expected)
