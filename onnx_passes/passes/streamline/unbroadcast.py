# ir.Value
import onnx_ir as ir

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# Base class for model transformations, un-broadcasting is easier to express as
# an instance of the base compared to pattern-based rules as this is more like a
# value-based rule
from onnx_passes.passes.base import Transformation

# NumPy used for calculations on shapes and constant tensors in rewrites and
# match conditions
import numpy as np

# Categorization of elementwise operator types
from onnx_passes.traits.elementwise import is_elementwise

# Naive un-broadcasting of NumPy arrays: Purely based on value repetition, does
# not consider other arrays/shapes participating in broadcasting
from onnx_passes.passes.util import unbroadcast, dropwhile


# Removes broadcastable dimensions from constant tensor inputs to elementwise
# operators to reduce the amount of redundant parameters
@passes.verify.equality
@passes.register("unbroadcast")
class UnbroadcastElementwise(Transformation):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Track whether any node actually changed
        modified = False

        # Tape recording initializers to be inserted into the graph at the end
        tape = ir.tape.Tape()

        # Get all constant tensors which should be lifted to initializers and
        # consider all those which are exclusively consumed by elementwise
        # operators for un-broadcasting
        for name, value in model.graph.initializers.items():
            for consumer in value.consumers():
                # Skip any non-elementwise operation as these usually do not
                # follow simple broadcasting semantics
                if not is_elementwise(consumer):
                    continue

                # Collect all input shapes to the consumer, except for the
                # currently handled input
                shapes = []

                # Bring back all dimensions which are contributed exclusively
                # by this input to not lose any dimensions
                for inp in consumer.inputs:
                    if inp != value:
                        shapes.append(inp.shape)

                # Cannot handle dynamic shapes here...
                if any(shape is None or shape.is_dynamic() for shape in shapes):
                    continue

                # NumPy array of the constant value as un-broadcasting is
                # implemented for NumPy arrays
                x = ir.convenience.get_const_tensor(value).numpy()
                # Unbroadcast the array if possible (otherwise the shape is
                # unchanged)
                y = unbroadcast(x)

                # Expected output shape after broadcasting all the original
                # inputs
                old_shape = np.broadcast_shapes(
                    x.shape, *(shape.numpy() for shape in shapes)
                )

                # Add padding to the unbroadcast shape to match the dimensions
                # of the original output shape
                shape = [*((len(old_shape) - len(y.shape)) * [1]), *y.shape]

                # Calculate the new broadcasting result after removing
                # dimensions from the input to see if any are lost
                new_shape = np.broadcast_shapes(
                    shape, *(shape.numpy() for shape in shapes)
                )

                # Restore some of the old dimensions if they would be lost after
                # un-broadcasting the input
                shape = np.where(
                    np.asarray(new_shape) != old_shape, old_shape, shape
                )

                # Restore values in these dimensions by broadcasting, i.e.,
                # replicating again
                y = np.broadcast_to(y, (*dropwhile(lambda s: s == 1, shape),))

                # Skip transforming if this yields the same shape as before...
                if y.shape == x.shape:
                    continue

                # Create a unique name derived from the original initializer as
                # well as the current consumer to avoid collisions
                unique_name = f"{value.name}_unbroadcast_{consumer.name}"

                # Create a new initializer registered for delayed insertion
                new_value = tape.initializer(ir.tensor(y), unique_name)

                # Might actually be consumed multiple times...
                for index, inp in enumerate(consumer.inputs):
                    if value == inp:
                        # Rewire the consumer to use the new initializer
                        consumer.replace_input_with(index, new_value)
                        # Mark the model as modified
                        modified = True

        # Insert all new initializers recorded by the tape
        for initializer in tape.initializers:
            model.graph.register_initializer(initializer)

        # Potentially modified model and indicator whether the model actually
        # changed
        return ir.passes.PassResult(model, modified)
