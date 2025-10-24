# ir.Model, ir.Value, ir.tape, ir.traversal, ir.passes
import onnx_ir as ir

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# Base class for deriving streamlining transformation passes
from onnx_passes.passes.base import Transformation

# Checks whether an ir.Value corresponds to a constant tensors, such as
# initializers, treatment of scalars can be simplified a lot
from onnx_passes.passes.util import is_constant, is_scalar

# Categorization of elementwise operator types
from onnx_passes.traits.elementwise import is_elementwise


# Transformation template matching elementwise n-ary operators followed by Split
# at the output: The order of split and elementwise operators can be switched
# by splitting the inputs to the n-ary operator.
#
# The template takes care of transferring attributes from the matched to the
# replacement elementwise operator.
#
# Note: In case of unary elementwise operators the match condition is trivially
# fulfilled and no extra Reshape is necessary.
@passes.verify.equality
@passes.register("reorder")
class MoveElementwisePastSplit(Transformation):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Track whether any node actually changed
        modified = False
        # Modify a deep copy of the original model
        model = ir.from_proto(ir.to_proto(model))

        # Records nodes and values to be inserted into the graph at the end
        tape = ir.tape.Tape()

        # Check each node in the graph (and subgraph) for elementwise-split
        # combinations
        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            # Skip if this is not a Split operator
            if (split := node).op_type != "Split":
                continue

            # Skip if the input is not produced by an elementwise operator
            if not is_elementwise((elementwise := split.inputs[0].producer())):
                continue

            # Count the number of constant inputs and reject the transformation
            # if this would result in more non-constant foldable splits
            # TODO: Consider dropping this restriction? Depends on how Expand
            #  operators will be treated in the long term...
            if sum(not is_constant(x) for x in elementwise.inputs) > 1:
                continue

            # Collect a list of ir.Values which are splits of each original
            # input to the elementwise operator
            splits = []

            # Expand (broadcast) and split each individual input of the
            # elementwise operator, i.e., move the Split in front of each input
            for x in elementwise.inputs:
                # Shortcut/Optimization for scalars: Skip expanding and
                # splitting, simply replicate the same value into each split
                if is_scalar(x):
                    # Replicate input value for each split
                    splits.append(len(split.outputs) * [x])
                    # Skip expanding and splitting
                    continue

                # Expand the input matching the broadcast shape of the
                # elementwise operation
                x = tape.op("Expand", [x, tape.op("Shape", [split.inputs[0]])])
                # Split the expanded input. The number of outputs must be made
                # explicit, otherwise will only match the first split.
                splits.append(tape.op_multi_out(
                    "Split", [x, split.inputs[1]], split.attributes,
                    num_outputs=len(split.outputs)
                ))

            # Insert elementwise operators consuming inputs splits for each of
            # the original splits
            for xs, out in zip(zip(*splits), split.outputs):
                # New elementwise operator following the split
                new = tape.op(elementwise.op_type, xs, elementwise.attributes)
                # Rewire all consumers of the original split to use the new one
                for consumer in out.consumers():
                    # Might actually be consumed multiple times...
                    for index, x in enumerate(consumer.inputs):
                        if out == x:
                            consumer.replace_input_with(index, new)

        # Only modify the graph if new nodes are recorded for insertion
        if tape.nodes:
            # Insert all collected nodes into the graph and ensure topological
            # sorting
            model.graph.extend(tape.nodes)
            model.graph.sort()
            # Mark the model as modified
            modified = True

        # Potentially modified copy of the model and indicator whether the model
        # actually changed
        return ir.passes.PassResult(model, modified)
