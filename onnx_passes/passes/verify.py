# ir.Model, ir.passes.PassResult, ir.from_proto, ir.to_proto, ...
import onnx_ir as ir
# NumPy for handling verification reference data
import numpy as np

# Base class for all custom ONNX IR passes developed in this library - this base
# class defines the (optional) interface for configuration and state tracking
from onnx_passes.passes.base import Pass, Analysis
# Utility functions on Pass objects for loading reference data and injecting
# pre- and post-conditions
from onnx_passes.passes.util import (
    inject_pre_post_condition, load_reference_data
)
# Custom, configurable wrapper around ONNX Runtime for model execution
from onnx_passes.passes.runtime import evaluate_model

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes


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
            return

        # Load reference input data for verification
        inputs, _ = load_reference_data(self)
        # Evaluate the model on the reference inputs and collect all results
        produced, context = evaluate_model(model, inputs,
                                           **self.config["onnxruntime"])
        # Set the produced output as the expectation checked against as the
        # post-condition
        self.expected = produced

    # Post-condition comparing model outputs to a reference for strict
    # equality - fails raising VerificationError if the output does not match
    def ensures(self: Pass, model: ir.Model) -> None:
        # Verification can be disabled globally by setting it to False or
        # specifying an explicitly empty configuration dictionary
        if not self.config.setdefault("verify", {True: True}):
            return

        # Load reference input data for verification
        inputs, _ = load_reference_data(self)
        # Evaluate the model on the reference inputs and collect all results
        produced, context = evaluate_model(model, inputs,
                                           **self.config["onnxruntime"])

        # Prepare logging verification results to the state dictionary
        self.state_dict.setdefault("verify", {})

        # Log the full input-output-expectation history for each
        # verification pass
        self.state_dict["verify"].setdefault("history", {})[self.id] = (
            inputs, produced, self.expected, context
        )

        # Compare for *strict* equality of *all* values from *all* outputs
        for output, x, y in zip(model.graph.outputs, produced, self.expected):
            if np.any(x != y):
                raise VerificationError(f"{output.name} not as expected")

        # Verbosity can be enabled globally by setting it to True
        self.config.setdefault("logging", {}).setdefault("verbose", False)
        # Verbosity should now be defined, either defaulting to False or
        # explicitly
        if self.config["logging"]["verbose"]:
            # TODO: Make use of a proper logger...
            print(f"Verified {self.__class__.__name__}")

    # Inject the pre- and post-condition into the ONNX IR pass and return the
    # modified class to allow for arbitrarily stacking decorators
    return inject_pre_post_condition(cls, requires, ensures)


# Explicit verification pass: Treated as an analysis pass, does not change the
# model but has side effects by running equality-based verification
@equality
@passes.register("verify")
class VerifyEquality(Analysis):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return ir.passes.PassResult(model, False)


# Injects tolerance-based verification into an ONNX IR pass by showing the model
# output on some reference to be within tolerance of the known expected output
def tolerance(cls: type[Pass]):
    # Pre-condition comparing model outputs to a reference for equality within
    # tolerance - should not fail, prepares for post-condition
    def requires(self: Pass, model: ir.Model) -> None:
        # Verification can be disabled globally by setting it to False or
        # specifying an explicitly empty configuration dictionary
        if not self.config.setdefault("verify", {True: True}):
            return

        # Load reference input data for verification
        inputs, _ = load_reference_data(self)
        # Evaluate the model on the reference inputs and collect all results
        produced, context = evaluate_model(model, inputs,
                                           **self.config["onnxruntime"])
        # Set the produced output as the expectation checked against as the
        # post-condition
        self.expected = produced

    # Post-condition comparing model outputs to a reference for equality within
    # tolerance - fails raising VerificationError if the output does not match
    def ensures(self: Pass, model: ir.Model) -> None:
        # Verification can be disabled globally by setting it to False or
        # specifying an explicitly empty configuration dictionary
        if not self.config.setdefault("verify", {True: True}):
            return

        # Load reference input data for verification
        inputs, _ = load_reference_data(self)
        # Evaluate the model on the reference inputs and collect all results
        produced, context = evaluate_model(model, inputs,
                                           **self.config["onnxruntime"])

        # Prepare logging of the error to the state dictionary to track model
        # degradation
        self.state_dict.setdefault("verify", {}).setdefault("max_abs_error", [])
        # Compute the maximum absolute error between produced and expected
        # output: Computing the mean, probably does not make sense...
        error = max_abs_error(produced, self.expected)
        # Append the error to the log associated to the just-verified pass
        self.state_dict["verify"]["max_abs_error"].append({cls.__name__: error})

        # Log the full input-output-expectation history for each
        # verification pass
        self.state_dict["verify"].setdefault("history", {})[self.id] = (
            inputs, produced, self.expected, context
        )

        # Read the optional verification tolerance configuration from the
        # configuration dictionary of the pass. Defaults according to NumPy.
        _tolerance = self.config["verify"].setdefault("tolerance", {})

        # Compare equality within tolerance of *all* values from *all* outputs
        for output, x, y in zip(model.graph.outputs, produced, self.expected):
            if not np.allclose(x, y, **_tolerance):
                raise VerificationError(f"{output.name} not within tolerance")

        # Verbosity can be enabled globally by setting it to True
        self.config.setdefault("logging", {}).setdefault("verbose", False)
        # Verbosity should now be defined, either defaulting to False or
        # explicitly
        if self.config["logging"]["verbose"]:
            # TODO: Make use of a proper logger...
            print(f"Verified {self.__class__.__name__}")

    # Inject the pre- and post-condition into the ONNX IR pass and return the
    # modified class to allow for arbitrarily stacking decorators
    return inject_pre_post_condition(cls, requires, ensures)


# Explicit verification pass: Treated as an analysis pass, does not change the
# model but has side effects by running tolerance-based verification
@tolerance
@passes.register("verify")
class VerifyTolerance(Analysis):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return ir.passes.PassResult(model, False)


# Injects metric-based verification into an ONNX IR pass by evaluating a metric,
# such as accuracy, over some reference dataset.
def metric(cls: type[Pass]):
    # Pre-condition comparing model outputs to a reference via a task-specific
    # metric - should not fail, prepares for post-condition
    def requires(self: Pass, model: ir.Model) -> None:
        # Verification can be disabled globally by setting it to False or
        # specifying an explicitly empty configuration dictionary
        if not self.config.setdefault("verify", {True: True}):
            return

        # Metric-based verification requires as section configuring how to
        # calculate metrics and the range of acceptable results: Assume empty...
        self.config["verify"].setdefault("metrics", [])

    # Post-condition comparing model outputs to a reference via a task-specific
    # metric - fails raising VerificationError if the output does not match
    def ensures(self: Pass, model: ir.Model) -> None:
        # Verification can be disabled globally by setting it to False or
        # specifying an explicitly empty configuration dictionary
        if not self.config.setdefault("verify", {True: True}):
            return

        # Load reference input and output data for verification
        inputs, expected = load_reference_data(self)

        # Specifiyng no expected reference output allows to skip all
        # metric-based verification
        if not expected:
            return

        # Evaluate the model on the reference inputs and collect all results
        produced, context = evaluate_model(model, inputs,
                                           **self.config["onnxruntime"])

        # Metric-based verification requires as section configuring how to
        # calculate metrics and the range of acceptable results
        for _metric, (_min, _max) in self.config["verify"]["metrics"].items():
            # The _metric key should be resolvable to some functions within
            # scope - custom functions can be injected via config "imports"
            function = eval(_metric)
            # Evaluate the metric on the produced vs. expected outputs
            value = function(produced, expected)

            # Prepare logging of the metric to the state dictionary to track
            # model degradation
            self.state_dict.setdefault("verify", {}).setdefault(_metric, [])
            # Append the metric to the log associated to the just-verified pass
            self.state_dict["verify"][_metric].append({cls.__name__: value})

            # Log the full input-output-expectation history for each
            # verification pass
            self.state_dict["verify"].setdefault("history", {})[self.id] = (
                inputs, produced, self.expected, context
            )

            # Assemble the potential error message in advance...
            msg = f"{_metric} {value} not within [{_min}, {_max}] as required"

            # Check whether the metric lies within the required range and raise
            # exception if not
            if not (_min <= value <= _max):
                raise VerificationError(msg)

        # Verbosity can be enabled globally by setting it to True
        self.config.setdefault("logging", {}).setdefault("verbose", False)
        # Verbosity should now be defined, either defaulting to False or
        # explicitly
        if self.config["logging"]["verbose"]:
            # TODO: Make use of a proper logger...
            print(f"Verified {self.__class__.__name__}")

    # Inject the pre- and post-condition into the ONNX IR pass and return the
    # modified class to allow for arbitrarily stacking decorators
    return inject_pre_post_condition(cls, requires, ensures)


# Explicit verification pass: Treated as an analysis pass, does not change the
# model but has side effects by running metric-based verification
@metric
@passes.register("verify")
class VerifyMetric(Analysis):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return ir.passes.PassResult(model, False)
