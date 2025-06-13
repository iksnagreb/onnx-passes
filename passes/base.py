# The base classes defined below are still not fully functional passes, but
# abstract bases themselves
from abc import ABC

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
