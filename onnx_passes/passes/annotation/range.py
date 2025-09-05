# ir.Model, ir.passes.PassResult, ir.from_proto, ir.to_proto, ...
import onnx_ir as ir

# Recursively iterate nodes from model graphs and function in order
from onnx_ir.traversal import RecursiveGraphIterator

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# Range propagation handlers are Callables and the exact type of range info is
# not known, i.e., can be of Any type
from typing import Callable, Any

# When iterating propagated output ranges, missing annotations are filled in via
# zipping with fallback values
from itertools import zip_longest

# Ranges are annotated with bounds as numpy arrays and propagate via np.minimum
# and np.maximum
import numpy as np

# Registry of per op-type range propagation functions
_registry = {}


# Registers a per op-type range propagation function
def register(op_type: str, domain: str = "", version: int | None = None):
    def inner(f: Callable):
        # Never overwrite handlers...
        assert (op_type, domain, version) not in _registry, (
            f"Range propagation for {(op_type, domain, version)} is already"
            f" registered, see {_registry[(op_type, domain, version)].__code__}"
        )
        # Add this function to the registry for the op-type
        _registry[(op_type, domain, version)] = f
        # Return the decorated function for chaining decorators
        return f

    # Return the wrapped inner decorator to be applied to the function decorated
    # by the outer decorator
    return inner


# Propagates the input range information to the outputs of the node forwarding
# inputs and attributes to a registered handler. Yields an empty list of range
# information if no suitable handler is registered.
def _propagate_range(node, *inputs, **attributes):
    # Fallback range propagation handler: Yields empty list of range information
    def _propagate_f(*_, **__):
        return []

    # First try to find a handler registered for the exact opset version falling
    # back to a version generic handler if available
    try:
        _propagate_f = _registry[(node.op_type, node.domain, node.version)]
    except KeyError:
        try:
            _propagate_f = _registry[(node.op_type, node.domain, None)]
        except KeyError:
            # Keep the fallback above
            pass

    # Forward all inputs and attributes to the range propagation handler
    return _propagate_f(*inputs, **attributes)


# Collects range information from an ir.Value taking existing annotations, data
# type bounds and actual value ranges (if value is constant) into account
def _get_range(value: ir.Value) -> tuple[Any, Any]:
    # First consider existing range annotation for the IR value form the
    # non-persistent metadata store
    try:
        _min, _max = value.meta["range"]
    except KeyError:
        _min, _max = None, None

    # Fallback to data type bounds if no range information is annotated so far
    try:
        _min = value.dtype.min if _min is None else _min
        _max = value.dtype.max if _max is None else _max
    except (TypeError, KeyError):
        pass

    # If the input is a constant tensor, also consider the
    # actual value range
    if (v := ir.convenience.get_const_tensor(value)) is not None:
        # Convert the input to numpy array format
        v = np.asarray(v.numpy())
        # Narrow down the range based on the actual values if possible, this
        # should result in a point interval but might also fail for some types
        try:
            _min = np.maximum(_min, v)
            _max = np.minimum(_max, v)
        except Exception:  # noqa: Whatever numpy raises...
            # Type probably does not support finding the minimum
            # or maximum, e.g. strings
            _min, _max = None, None

    # Wrap minimum and maximum as numpy arrays if available
    _min = np.asarray(_min) if _min is not None else None
    _max = np.asarray(_max) if _max is not None else None

    # Range information: Bounds of the IR value or None if no bounds could be
    # determined
    return _min, _max


# Annotates range information on an ir.Value falling back to data type bounds
# if no information is given
def _set_range(value: ir.Value, _min: Any = None, _max: Any = None):
    # Fallback to data type bounds if no range information is give
    try:
        _min = value.dtype.min if _min is None else _min
        _max = value.dtype.max if _max is None else _max
    except (TypeError, KeyError):
        pass

    # Wrap minimum and maximum as numpy arrays if available
    _min = np.asarray(_min) if _min is not None else None
    _max = np.asarray(_max) if _max is not None else None

    # Annotate the intermediate metadata store of the IR value, this will not be
    # serialized
    # TODO: This also means the information will be lost after copying the model
    #  by serializing/deserializing to/from protobuf...
    value.meta["range"] = (_min, _max)


# Annotates reachable (i.e., consumed/produced by some node) value infos with
# additional range information. These are derived from data type bounds or
# actual value ranges for constant values and propagate if a range propagation
# handler is registered for the operator type of the producer node.
@passes.register("range-annotation")
class RangeAnnotation(passes.base.Annotation):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Modify a deep copy of the original model as Annotation passes are
        # functional passes...
        model = ir.from_proto(ir.to_proto(model))
        # Iterate all nodes in the graph - forward iterator as we are
        # propagating ranges from the input to the output
        for node in RecursiveGraphIterator(model.graph, reverse=False):
            # Collect input ranges to this node, None marks missing optional
            # inputs, missing optional trailing inputs can also be omitted
            inputs: list[tuple | None] = [None for _ in node.inputs]
            # Enumerate all inputs to the node
            for i, inp in enumerate(node.inputs):
                # Optional inputs can be represented by None...
                if inp is not None:
                    # Collect input range information
                    _min, _max = _get_range(inp)
                    # Annotate in case this returned fallback bounds
                    _set_range(inp, _min, _max)
                    # Update minimum and maximum for range propagation
                    inputs[i] = (inp, _min, _max)
            # Propagate the input range information to the outputs
            outputs = _propagate_range(node, *inputs, **node.attributes)

            # Pair up all outputs of the node with propagated range information
            # and set/update the value metadata annotation
            for out, (_min, _max) in (
                    zip_longest(node.outputs, outputs, fillvalue=(None, None))):
                _set_range(out, _min, _max)

        # Annotation passes do not modify the model, but might add metadata or
        # fill the state dictionary (not considered as modified)
        return ir.passes.PassResult(model, modified=True)


@register("Identity")
def _propagate_range_identity(x):
    return x


# TODO: Simplify: _min/_max must be scalars according to ONNX reference... but
#  this could also be reused for elementwise minimum/maximum?
@register("Clip")
def _propagate_range_clip(x, _min=None, _max=None):
    # If the input x is not present, something went wrong, probably the model is
    # in an illegal state. To be safe, make no assumptions on the output range.
    if x is None:
        return []

    # Clipping bounds are optional and fall back to the data type bounds
    _min = (None, None, None) if _min is None else _min
    _max = (None, None, None) if _max is None else _max

    def default_min(v):
        return v if v is not None else np.asarray(x[0].dtype.min)

    def default_max(v):
        return v if v is not None else np.asarray(x[0].dtype.max)

    # The output range if clipping is the most restricted of the inputs, i.e.,
    # the maximum of the lower bounds and the minimum of the upper bounds
    _min = np.maximum(default_min(x[1]), default_min(_min[1]))
    _max = np.minimum(default_max(x[2]), default_max(_max[2]))

    # Range of the single output of the clip operator
    return [(np.asarray(_min), np.asarray(_max))]
