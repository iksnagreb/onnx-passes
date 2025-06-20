# Base classes for all custom ONNX IR passes developed in this library
from onnx_passes.passes.base import Pass

# Registry of ONNX IR passes by names/categories
_registry = {}


# Registers an ONNX IR pass with a name/category
def register(name: str):
    # Inner decorator registering the pass
    def inner(cls: type):
        # Must be derived from the ONNX IR pass base
        assert issubclass(cls, Pass)
        # Add this transformation to the registry
        _registry.setdefault(name, []).append(cls)
        # Return the decorated class for chaining decorators
        return cls

    # Return the wrapped inner decorator to be applied to the type decorated by
    # the outer decorator
    return inner


# Collects passes by names/categories
def collect(names: list[str | type]):
    # Resolves pass classes from type or string indexing the registry
    def _resolve(key: str | type):
        # If the key is a type, this is already the resolved pass class
        if isinstance(key, type):
            return [key]
        # Otherwise flatten and resolve the pass class from the registry
        return [cls for cls in _registry[key]]

    # Flatten and resolve all pass classes from the pass registry
    return [cls for key in names for cls in _resolve(key)]


# Infrastructure for injecting automatic verification of passes via pre- and
# post-conditions evaluated by a Passmanager
import onnx_passes.passes.verify
# Composing sequences of passes managed by nested PassManager instances and
# potential exhaustive application
import onnx_passes.passes.compose

# Some passes should always be available by default without extra, dynamic
# imports, such as cleanup, checks and verification related passes.
from onnx_passes.passes import analysis, annotation, convert, constants, \
    cleanup, inline, streamline
