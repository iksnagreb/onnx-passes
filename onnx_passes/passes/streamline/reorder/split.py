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
# fulfilled and no extra broadcasting is necessary.
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


# What is implemented here is basically the same as MoveElementwisePastConcat,
# so it makes sense to reuse the same concept of two operations being the same
from onnx_passes.passes.streamline.reorder.concat import _is_same_op

# Sequence collection type annotation (~ generic list annotation)
from typing import Sequence


# Streamlining pass undoing the more general streamlining of elementwise past
# split operations: Sometimes elementwise operations are stuck following the
# Split, and it could be more efficient to have a single one in front.
@passes.verify.equality
@passes.register()
class MoveSplitPastElementwise(Transformation):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Track whether any node actually changed
        modified = False
        # Modify a deep copy of the original model
        model = ir.from_proto(ir.to_proto(model))

        # Records nodes and values to be inserted into the graph at the end
        tape = ir.tape.Tape()

        def op_like(other: ir.Node, x: Sequence[ir.Value | None]) -> ir.Value:
            return tape.op(other.op_type, x, other.attributes)

        # Check each node in the graph (and subgraph) for elementwise-split
        # combinations
        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            # Skip if this is not a Split operator
            if (split := node).op_type != "Split":
                continue

            # If there is no producer of this input, it is probably a global
            # input of the graph, which needs special treatment
            if not (inp := split.inputs[0]).producer():
                # If this is not a global input, this is an initializer, which
                # should be handled somewhere else
                if not inp.is_graph_input():
                    continue

                # New tensor connecting the global input to the
                # auxiliary identity inserted after the concat
                new = ir.Value(type=inp.type, shape=inp.shape)
                # For global inputs we can plug in the identity between to
                # pretend to have producers for handling all cases below
                new = tape.op("Identity", [inp], output=new)
                # Rewire the split input to use the new instead of the old
                split.replace_input_with(0, new)

            # Collect consumers at each outputs forming a list of lists of
            # candidate operators to be handled
            consumers = [x.consumers() for x in split.outputs]

            # Handling multiple (different) consumers of the same output is not
            # possible, as these cannot be concatenated into a single operator
            if any((len(ops) > 1 for ops in consumers)):
                continue

            # Flatten the consumer list to get rid of the nesting (by now there
            # is at most one consumer per output)
            consumers = [ops[0] for ops in consumers]

            # Loop until done (rather complex condition deciding when none of
            # the operators can be moved)
            while any(is_elementwise(op) for op in consumers):
                # Find the first consumer which is an elementwise operator (we
                # cannot filter this in advance so we can keep track of the
                # top-level inputs of this pattern)
                elementwise = None
                for op in consumers:
                    if is_elementwise(op):
                        elementwise = op
                        break

                # If all operators are identical and elementwise, all their
                # inputs can be concatenated and moved all at once
                if all(_is_same_op(op, elementwise) for op in consumers):
                    # Broadcast all real (there are no auxiliaries) inputs to
                    # the output shape
                    inputs = []
                    # Each consumer can be treated the same as these are all the
                    # same type
                    for op in consumers:
                        # Collect list of all expanded inputs for this operator
                        xs = []
                        # Expand each input to the elementwise operation to the
                        # shape of the single output of the operation
                        for x in op.inputs:
                            if x.producer() == split:
                                xs.append(x)
                            else:
                                xs.append(
                                    tape.op(
                                        "Expand", [
                                            x, tape.op(
                                                "Shape", [
                                                    op_like(op, op.inputs)
                                                ]
                                            )
                                        ]
                                    )
                                )
                        # Add to list of lists of expanded inputs, this list is
                        # built up in transposed order, i.e., the first entries
                        # from each list will be concatenated, the second, ...
                        inputs.append(xs)

                    # Concatenate all matched-up inputs and forward as inputs to
                    # the new instance of the elementwise operation
                    concatenated = []
                    # Zipping the unpacked outermost list transposes the inputs,
                    # i.e., each inner list now represents of tuple of inputs to
                    # be concatenated
                    for xs in zip(*inputs):
                        # If this comes from the original split, just rewire to
                        # skip the split, no Concat needed as it is reversed
                        if all(x.producer() == split for x in xs):
                            concatenated.append(split.inputs[0])
                        # Concatenate all other inputs along the same axis as
                        # the split
                        else:
                            concatenated.append(
                                tape.op("Concat", xs, split.attributes)
                            )
                    # Create the new instance of the elementwise operator
                    # receiving the concatenated inputs
                    new = op_like(elementwise, concatenated)

                    # Create a new split operation following the expanded and
                    # concatenated elementwise operation
                    splits = tape.op_multi_out(
                        "Split", [new, split.inputs[1]], split.attributes,
                        num_outputs=len(split.outputs)
                    )

                    # Wire up the new operator to the old output and remember
                    # new inputs for wiring up the producers
                    for op, new in zip(consumers, splits):
                        for consumer in op.outputs[0].consumers():
                            # Might actually be consumed multiple times...
                            for i, x in enumerate(consumer.inputs):
                                if op.outputs[0] == x:
                                    consumer.replace_input_with(i, new)

                    # Continue with next candidate operator, there might be up
                    # to the number of split outputs different operations
                    break

                # TODO: Implement auxiliary identity insertion in case not all
                #  consumers are the same, see MoveElementwisePastConcat

        # Insert all new initializers recorded by the tape
        for initializer in tape.initializers:
            # Register the initializer ir.Value recorded by the tape
            model.graph.register_initializer(initializer)
            # Mark the model as modified
            modified = True

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
