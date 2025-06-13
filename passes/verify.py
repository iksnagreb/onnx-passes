# ir.Model, ir.passes.PassResult, ir.from_proto, ir.to_proto, ...
import onnx_ir as ir
# NumPy for handling verification reference data
import numpy as np

# Base class for all custom ONNX IR passes developed in this library - this base
# class defines the (optional) interface for configuration and state tracking
from passes.base import Pass
# Utility functions on Pass objects for loading reference data and injecting
# pre- and post-conditions
from passes.util import inject_pre_post_condition, load_reference_data
# Custom, configurable wrapper around ONNX Runtime for model execution
from passes.runtime import evaluate_model


# Exception type indicating verification failure while evaluating pre- and
# post-conditions - currently does not do add anything ontop the base Exception.
class VerificationError(Exception):
    ...


# Calculates the maximum absolute error between all outputs and expected outputs
def max_abs_error(produced: list, expected: list) -> float:
    return max(np.max(np.abs(x - y)) for x, y in zip(produced, expected))


# Injects equality-based verification into an ONNX IR pass by checking if the
# model output on some reference is equal to the known expected output
def equality(cls: type[Pass]):
    # Pre-condition comparing model outputs to a reference for strict
    # equality - should not fail, prepares for post-condition
    def requires(self: Pass, model: ir.Model) -> None:
        # Verification can be disabled globally by setting it to False or
        # specifying an explicitly empty configuration dictionary
        if not self.config.setdefault("verify", {True: True}):
            # Just exit here doing nothing...
            return

        # Load reference input data for verification
        inputs, _ = load_reference_data(self)
        # Evaluate the model on the reference inputs and collect all results
        produced = evaluate_model(model, inputs)
        # Set the produced output as the expectation checked against as the
        # post-condition
        self.expected = produced

    # Post-condition comparing model outputs to a reference for strict
    # equality - fails raising VerificationError if the output does not match
    def ensures(self: Pass, model: ir.Model) -> None:
        # Verification can be disabled globally by setting it to False or
        # specifying an explicitly empty configuration dictionary
        if not self.config.setdefault("verify", {True: True}):
            # Just exit here doing nothing...
            return

        # Load reference input data for verification
        inputs, _ = load_reference_data(self)
        # Evaluate the model on the reference inputs and collect all results
        produced = evaluate_model(model, inputs)

        # Compare for *strict* equality of *all* values from *all* outputs
        for output, x, y in zip(model.graph.outputs, produced, self.expected):
            if np.any(x != y):
                raise VerificationError(f"{output.name} not as expected")

    # Inject the pre- and post-condition into the ONNX IR pass and return the
    # modified class to allow for arbitrarily stacking decorators
    return inject_pre_post_condition(cls, requires, ensures)


# Injects tolerance-based verification into an ONNX IR pass by showing the model
# output on some reference to be within tolerance of the known expected output
def tolerance(cls: type[Pass]):
    # Pre-condition comparing model outputs to a reference for equality within
    # tolerance - fails raising VerificationError if the output does not match
    def requires(self: Pass, model: ir.Model) -> None:
        # Verification can be disabled globally by setting it to False or
        # specifying an explicitly empty configuration dictionary
        if not self.config.setdefault("verify", {True: True}):
            # Just exit here doing nothing...
            return

        # Load reference input data for verification
        inputs, _ = load_reference_data(self)
        # Evaluate the model on the reference inputs and collect all results
        produced = evaluate_model(model, inputs)
        # Set the produced output as the expectation checked against as the
        # post-condition
        self.expected = produced

    # Post-condition comparing model outputs to a reference for equality within
    # tolerance - fails raising VerificationError if the output does not match
    def ensures(self: Pass, model: ir.Model) -> None:
        # Verification can be disabled globally by setting it to False or
        # specifying an explicitly empty configuration dictionary
        if not self.config.setdefault("verify", {True: True}):
            # Just exit here doing nothing...
            return

        # Load reference input data for verification
        inputs, _ = load_reference_data(self)
        # Evaluate the model on the reference inputs and collect all results
        produced = evaluate_model(model, inputs)

        # Prepare logging of the error to the state dictionary to track model
        # degradation
        self.state_dict.setdefault("verify", {}).setdefault("max_abs_error", [])
        # Compute the maximum absolute error between produced and expected
        # output: Computing the mean, probably does not make sense...
        error = max_abs_error(produced, self.expected)
        # Append the error to the log associated to the just-verified pass
        self.state_dict["verify"]["max_abs_error"].append({cls.__name__: error})


        # Read the optional verification tolerance configuration from the
        # configuration dictionary of the pass. Defaults according to NumPy.
        _tolerance = self.config["verify"].setdefault("tolerance", {})

        # Compare equality within tolerance of *all* values from *all* outputs
        for output, x, y in zip(model.graph.outputs, produced, self.expected):
            if not np.allclose(x, y, **_tolerance):
                raise VerificationError(f"{output.name} not within tolerance")

    # Inject the pre- and post-condition into the ONNX IR pass and return the
    # modified class to allow for arbitrarily stacking decorators
    return inject_pre_post_condition(cls, requires, ensures)


# Injects metric-based verification into an ONNX IR pass by evaluating a metric,
# such as accuracy, over some reference dataset.
#
# Once the model passes metric-based verification, the generated output is used
# as a new reference for following equality- or tolerance-based verification.
def metric(cls: type[Pass]):
    # Pre-condition comparing model outputs to a reference via a task-specific
    # metric - fails raising VerificationError if the output does not match
    def requires(self: Pass, model: ir.Model) -> None:
        # Verification can be disabled globally by setting it to False or
        # specifying an explicitly empty configuration dictionary
        if not self.config.setdefault("verify", {True: True}):
            # Just exit here doing nothing...
            return

        # TODO: Implement the actual verification here...
        print(f"Verifying by metric before {cls.__name__}")

    # Post-condition comparing model outputs to a reference via a task-specific
    # metric - fails raising VerificationError if the output does not match
    def ensures(self: Pass, model: ir.Model) -> None:
        # Verification can be disabled globally by setting it to False or
        # specifying an explicitly empty configuration dictionary
        if not self.config.setdefault("verify", {True: True}):
            # Just exit here doing nothing...
            return

        # TODO: Implement the actual verification here...
        print(f"Verifying by metric after {cls.__name__}")

    # Inject the pre- and post-condition into the ONNX IR pass and return the
    # modified class to allow for arbitrarily stacking decorators
    return inject_pre_post_condition(cls, requires, ensures)
