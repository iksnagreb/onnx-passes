# ONNX IR passes base classes
import passes.base

# Registry of ONNX IR passes by names/categories
_registry = {}


# Registers an ONNX IR pass with a name/category
def register(name: str):
    # Inner decorator registering the pass
    def inner(cls: type):
        # Must be derived from the ONNX IR pass base
        assert issubclass(cls, passes.base.Pass)
        # Add this transformation to the registry
        _registry.setdefault(name, []).append(cls)
        # Return the decorated class for chaining decorators
        return cls

    # Return the wrapped inner decorator to be applied to the type decorated by
    # the outer decorator
    return inner


# Collects passes by names/categories
def collect(names: list[str]):
    # Flatten all passes registered for names
    return [cls for name in names for cls in _registry[name]]
