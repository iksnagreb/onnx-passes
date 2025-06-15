# os.makedirs for creating logging directories on the fly
import os

# The base classes defined below are still not fully functional passes, but
# abstract bases themselves
from abc import ABC

# ir.Model, ir.save, ...
import onnx_ir as ir

# Base classes inherited from ONNX IR used by the custom ONNX passes
from onnx_ir.passes import PassBase, FunctionalPass


# Base class for deriving all custom passes of the ONNX IR pass library: This
# adds configuration and state handling and serves as a marker type for building
# the registry of named/categorized passes.
class Pass(PassBase, ABC):
    # Initializes a pass sets references to optional configuration and state
    # dictionary as instance attributes
    def __init__(self, config: dict | None, state: dict | None):
        self.config = config
        self.state_dict = state
        # Used by verification to inject expected outputs for post-condition
        self.expected = None
        self._id = None

    # Inject generating a unique pass-id available for all pre- and post-
    # conditions, as well as the wrapped __call__ and call methods
    def __call__(self, *args, **kwargs):
        # Count the number of passes already applied to the model to derive
        # unique checkpoint filenames
        i = len(self.state_dict.setdefault("history", []))
        # Generate a unique pass id valid until the next call to this pass
        self._id = f"{i:08d}-{type(self).__name__}"
        # Now forward all arguments to the base-class __call__ implementation
        return PassBase.__call__(self, *args, **kwargs)  # noqa: *args, *kwargs

    # Unique pass-id to identify a pass across repeated applications within a
    # sequence of passes
    @property
    def id(self):
        return self._id

    # Pre-condition evaluated before entering a pass - implements verbosity
    def requires(self, model: ir.Model) -> None:
        # Verbosity can be enabled globally by setting it to True
        self.config.setdefault("logging", {}).setdefault("verbose", False)
        # Verbosity should now be defined, either defaulting to False or
        # explicitly
        if self.config["logging"]["verbose"]:
            # TODO: Make use of a proper logger...
            print(f"Entering {self.__class__.__name__}")

        # Model checkpointing can be disabled globally by setting the option to
        # False, otherwise it is interpreted as a filename to write the model
        # checkpoint to
        if self.config["logging"].setdefault("checkpoint", False):
            # Mark this as the before-the-pass checkpoint
            filename = f"before-{self.config['logging']['checkpoint']}"
            # Save the model checkpoint
            ir.save(model, filename)

    # Post-condition evaluated after leaving a pass - implements verbosity
    def ensures(self, model: ir.Model) -> None:
        # Verbosity can be enabled globally by setting it to True
        self.config.setdefault("logging", {}).setdefault("verbose", False)
        # Verbosity should now be defined, either defaulting to False or
        # explicitly
        if self.config["logging"]["verbose"]:
            # TODO: Make use of a proper logger...
            print(f"Leaving {self.__class__.__name__}")

        # Model checkpointing can be disabled globally by setting the option to
        # False, otherwise it is interpreted as a filename to write the model
        # checkpoint to
        if self.config["logging"].setdefault("checkpoint", False):
            # Mark this as the after-the-pass checkpoint
            filename = f"after-{self.config['logging']['checkpoint']}"
            # Save the model checkpoint
            ir.save(model, filename)

        # Detailed logging of all intermediate models can be disabled globally
        # by setting the option to False, otherwise it is interpreted as a
        # pathname to write the models checkpoints to
        if self.config["logging"].setdefault("keep_intermediates", False):
            # Get the logging directory pathname
            path = self.config["logging"]["keep_intermediates"]
            # Make sure the directory exists...
            os.makedirs(path, exist_ok=True)
            # Mark this as the after-the-pass checkpoint
            filename = os.path.join(path, f"{self.id}.onnx")
            # Save the model checkpoint
            ir.save(model, filename)

        # Write a detailed history of passes finished on the model into the
        # state dictionary
        self.state_dict.setdefault("history", []).append(type(self))


# Base class for deriving analysis passes, which are side-effect-only passes,
# i.e., may only modify configuration and state dictionaries or other externally
# referenced objects (this includes printing/output), but not the model.
class Analysis(Pass, ABC):
    @property
    def in_place(self) -> bool:
        return True

    @property
    def changes_input(self) -> bool:
        return False


# Base class for deriving annotation passes, which are functional passes, i.e.,
# may return a modified copy of the original model but may not modify the
# original model. Annotation passes *should* not modify the structure or any
# values contained in the model, only attributes, shapes or data types.
class Annotation(Pass, FunctionalPass, ABC):
    ...


# Base class for deriving transformation passes, which are functional passes,
# i.e., may return a modified copy of the original model but may not modify the
# original model. Transformation passes may modify arbitrary properties of the
# model, including structure and values.
class Transformation(Pass, FunctionalPass, ABC):
    ...
