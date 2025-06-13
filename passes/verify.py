# ir.Model, ir.passes.PassResult, ir.from_proto, ir.to_proto, ...
import onnx_ir as ir

# Base class for all custom ONNX IR passes developed in this library - this base
# class defines the (optional) interface for configuration and state tracking
from passes.base import Pass


# Exception type indicating verification failure while evaluating pre- and
# post-conditions - currently does not do add anything ontop the base Exception.
class VerificationError(Exception):
    ...


# Injects pre- and post-condition methods into an ONNX IR pass, i.e., wraps and
# overwrites the .requires and .ensures methods.
def _inject_pre_post_condition(cls: type[Pass], pre: callable, post: callable):
    # The wrapped pass might already have pre- and post-conditions defined which
    # we should preserve, adding the verification on top...
    _requires, _ensures = cls.requires, cls.ensures

    # Evaluate the new followed by the original pre-condition - we do this
    # afterward to preserve the order of operations when stacking decorators
    def requires(self: Pass, model: ir.Model) -> None:
        pre(self, model), _requires(self, model)

    # Evaluate the original followed by the new post-condition - we do this
    # first to preserve the order of operations when stacking decorators
    def ensures(self: Pass, model: ir.Model) -> None:
        _ensures(self, model), post(self, model)

    # Inject the new pre- and post-condition methods overwriting the exiting
    # methods which have been wrapped by the new ones.
    cls.requires, cls.ensures = requires, ensures
    # Return the modified class
    return cls


# Injects equality-based verification into an ONNX IR pass by showing the model
# output on some reference to be equal to the known expected output
def equality(cls: type[Pass]):
    # Define a pre-condition comparing model outputs to a reference for strict
    # equality - fails raising VerificationError if the output does not match
    def requires(self: Pass, model: ir.Model) -> None:
        # Verification can be disabled globally by setting it to False or
        # specifying an explicitly empty configuration dictionary
        if not self.config.setdefault("verify", True):
            # Just exit here doing nothing...
            return

        # TODO: Implement the actual verification here...
        print(f"Verifying for equality before {cls.__name__}")

    # Define a post-condition comparing model outputs to a reference for strict
    # equality - fails raising VerificationError if the output does not match
    def ensures(self: Pass, model: ir.Model) -> None:
        # Verification can be disabled globally by setting it to False or
        # specifying an explicitly empty configuration dictionary
        if not self.config.setdefault("verify", True):
            # Just exit here doing nothing...
            return

        # TODO: Implement the actual verification here...
        print(f"Verifying for equality after {cls.__name__}")

    # Inject the pre- and post-condition into the ONNX IR pass and return the
    # modified class to allow for arbitrarily stacking decorators
    return _inject_pre_post_condition(cls, requires, ensures)


# Injects tolerance-based verification into an ONNX IR pass by showing the model
# output on some reference to be within tolerance of the known expected output
def tolerance(cls: type[Pass]):
    # Pre-condition comparing model outputs to a reference for equality within
    # tolerance - fails raising VerificationError if the output does not match
    def requires(self: Pass, model: ir.Model) -> None:
        # Verification can be disabled globally by setting it to False or
        # specifying an explicitly empty configuration dictionary
        if not self.config.setdefault("verify", True):
            # Just exit here doing nothing...
            return

        # TODO: Implement the actual verification here...
        print(f"Verifying for equality within tolerance before {cls.__name__}")

    # Post-condition comparing model outputs to a reference for equality within
    # tolerance - fails raising VerificationError if the output does not match
    def ensures(self: Pass, model: ir.Model) -> None:
        # Verification can be disabled globally by setting it to False or
        # specifying an explicitly empty configuration dictionary
        if not self.config.setdefault("verify", True):
            # Just exit here doing nothing...
            return

        # TODO: Implement the actual verification here...
        print(f"Verifying for equality within tolerance after {cls.__name__}")

    # Inject the pre- and post-condition into the ONNX IR pass and return the
    # modified class to allow for arbitrarily stacking decorators
    return _inject_pre_post_condition(cls, requires, ensures)


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
        if not self.config.setdefault("verify", True):
            # Just exit here doing nothing...
            return

        # TODO: Implement the actual verification here...
        print(f"Verifying by metric before {cls.__name__}")

    # Post-condition comparing model outputs to a reference via a task-specific
    # metric - fails raising VerificationError if the output does not match
    def ensures(self: Pass, model: ir.Model) -> None:
        # Verification can be disabled globally by setting it to False or
        # specifying an explicitly empty configuration dictionary
        if not self.config.setdefault("verify", True):
            # Just exit here doing nothing...
            return

        # TODO: Implement the actual verification here...
        print(f"Verifying by metric after {cls.__name__}")

    # Inject the pre- and post-condition into the ONNX IR pass and return the
    # modified class to allow for arbitrarily stacking decorators
    return _inject_pre_post_condition(cls, requires, ensures)
