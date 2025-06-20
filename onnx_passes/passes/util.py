# ir.Model
import onnx_ir as ir
# np.load for loading reference data
import numpy as np

# Base class for all custom ONNX IR passes developed in this library - this base
# class defines the (optional) interface for configuration and state tracking
from onnx_passes.passes.base import Pass


# Injects pre- and post-condition methods into an ONNX IR pass, i.e., wraps and
# overwrites the .requires and .ensures methods.
def inject_pre_post_condition(cls: type[Pass], pre: callable, post: callable):
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


# Loads reference data from the config or state dictionary of an ONNX IR pass by
# first considering the state dictionary
def load_reference_data(p: Pass) -> tuple[list, list]:
    # Accessing non-existing dictionaries might result in AttributeError or
    # TypeError
    try:
        # First try the state dictionary if it contains a reference section
        if p.state_dict and "reference" in p.state_dict:
            return (p.state_dict["reference"].setdefault("inp", []),
                    p.state_dict["reference"].setdefault("out", []))

        # Make sure the next test does not result in KeyError or ValueError by
        # injecting empty default lists
        p.config["reference"].setdefault("inp", [])
        p.config["reference"].setdefault("out", [])

        # If no reference data is tracked via the state dictionary, this is probably
        # the first attempt at loading the data: Check the config
        if p.config and "reference" in p.config:
            return ([np.load(file) for file in p.config["reference"]["inp"]],
                    [np.load(file) for file in p.config["reference"]["out"]])

        # Nothing found, return two empty lists indicating no inputs/outputs
        return [], []
    # If the "references" section is not present, we might end up here
    except (AttributeError, TypeError):
        return [], []
