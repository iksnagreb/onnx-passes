# ir.Model, ir.Value, ir.tape, ir.traversal, ir.passes
import onnx_ir as ir

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# Base class for deriving streamlining transformation passes
from onnx_passes.passes.base import Transformation

# Categorization of elementwise operator types
from onnx_passes.traits.elementwise import is_elementwise


def _is_same_op(other: ir.Node | None, op: ir.Node | None) -> bool:
    # Trivial cases: Same if both are None or both are the identical object
    if other is None and op is None or other == op:
        return True

    # Trivial case: Not same if one operation is None while the other is not
    if op is None or other is None:
        return False

    # Now these are two different nodes which still might represent the same
    # type of operation if, they identify the same (domain, op-type, overload),
    if other.op_identifier() == op.op_identifier():
        # The operator version is the same (some notion of compatibility could
        # be enough?)
        if other.version == op.version:
            # The configuration, i.e., the node attributes, are the same
            if other.attributes == op.attributes:
                # The number of inputs must be the same
                return len(other.inputs) == len(op.inputs)

    # Nodes represent different operations
    return False


def _identity(tape: ir.tape.Tape, op: ir.Node, x: ir.Value) -> ir.Value | None:
    # Mapping of identity operation inputs for operator types - for now only
    # implements the basic arithmetic operators.
    identities = {
        "Add": [x, 0], "Sub": [x, 0], "Mul": [x, 1], "Div": [1, x]
    }

    # If there is an identity for this operator type registered, create the
    # identity operator after turning all constants into constant operators
    if op.op_type in identities:
        for index, v in enumerate(identities[op.op_type]):
            # At least one input to the identity will be the input x, all others
            # might be scalars which need to be turned into  constant operators
            if not isinstance(v, ir.Value):
                identities[op.op_type][index] = tape.op(
                    "CastLike", [tape.op("Constant", [], {"value_int": v}), x]
                )

        # Elementwise operator of the same type but identity inputs
        return tape.op(op.op_type, identities[op.op_type], op.attributes)

    # No identity implemented for the operator represented by the node
    return None


# Transformation matching elementwise followed by Concat at the output: The
# order of concat and elementwise operators can be switched by concatenating the
# inputs to the elementwise operator.
#
# Note: As this manual rewiring of the graph tends to be messy, this should be
# applied as part of an exhaustive composition followed by constant-folding and
# cleanup transformations until the graph stops changing, such as streamline.
@passes.verify.equality
@passes.register("reorder")
class MoveElementwisePastConcat(Transformation):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Track whether any node actually changed
        modified = False

        # Records nodes and values to be inserted into the graph at the end
        tape = ir.tape.Tape()

        def op_like(other: ir.Node, x: list[ir.Value | None]) -> ir.Value:
            return tape.op(other.op_type, x, other.attributes)

        # Check each node in the graph (and subgraph) for elementwise-concat
        # combinations
        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            if (concat := node).op_type != "Concat":
                continue

            # If there are no consumers of this output, it is probably a global
            # output of the graph, which needs special treatment
            if not (out := concat.outputs[0]).consumers():
                # If this is not a global output, this is probably unused, which
                # should be handled somewhere else
                if not out.is_graph_output():
                    continue

                # New output tensor connecting the global output to the
                # auxiliary identity inserted after the concat
                output = ir.Value(type=out.type, shape=out.shape)
                # For global inputs we can plug in the identity between to
                # pretend to have consumers for handling all cases below
                output = tape.op("Identity", [out], output=output)
                # Rewire the graph outputs to use the new instead of the old
                model.graph.outputs[model.graph.outputs.index(out)] = output

            # Collect producers at each input forming a list of candidate
            # operators to be handled
            producers = [x.producer() for x in concat.inputs]

            # Loop until done (rather complex condition deciding when none of
            # the operators can be moved)
            while any(is_elementwise(op) for op in producers):
                # Find the first producer which is an elementwise operator (we
                # cannot filter this in advance so we can keep track of the
                # top-level inputs of this pattern)
                elementwise = None
                for op in producers:
                    if is_elementwise(op):
                        elementwise = op
                        break

                # If all operators are identical and elementwise, all their
                # inputs can be concatenated and moved all at once
                if all(_is_same_op(op, elementwise) for op in producers):
                    # Broadcast all real (there are no auxiliaries) inputs to
                    # the output shape
                    inputs = []
                    # Each producer can be treated the same as these are all the
                    # same type
                    for op in producers:
                        # Collect list of all expanded inputs for this operator
                        xs = []
                        # Expand each input to the elementwise operation to the
                        # shape of the single output of the operation
                        for x in op.inputs:
                            xs.append(
                                tape.op(
                                    "Expand", [
                                        x, tape.op(
                                            "Shape", [
                                                op.outputs[0]
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
                    # the new instance of the producer operation
                    concatenated = []
                    # Zipping the unpacked outermost list transposes the inputs,
                    # i.e., each inner list now represents of tuple of inputs to
                    # be concatenated
                    for xs in zip(*inputs):
                        concatenated.append(
                            tape.op("Concat", xs, concat.attributes)
                        )
                    # Create the new instance of the elementwise operator
                    # receiving the concatenated inputs
                    new = op_like(elementwise, concatenated)

                    # Wire up the new operator to the old output and remember
                    # new inputs for wiring up the producers
                    for consumer in concat.outputs[0].consumers():
                        # Might actually be consumed multiple times...
                        for i, x in enumerate(consumer.inputs):
                            if concat.outputs[0] == x:
                                consumer.replace_input_with(i, new)

                    # Continue with next candidate operator, there might be up
                    # to the number of concat inputs different operations
                    break

                # If not all operators are identical, find the first one which
                # provides an identity to insert as auxiliary operators
                elementwise = None
                for op in producers:
                    if (_identity(tape, op, op.inputs[0])) is not None:
                        if is_elementwise(op):
                            elementwise = op
                            break

                # If we cannot insert any auxiliary identities, we are done
                # moving operators
                if elementwise is None:
                    break

                # For each operator which is not the same as this operator,
                # insert a new auxiliary identity into the graph
                for index, inp in enumerate(concat.inputs):
                    if not _is_same_op(inp.producer(), elementwise):
                        # Auxiliary elementwise operation of the same type and
                        # attribute configuration but using identity inputs
                        new = _identity(tape, elementwise, inp)

                        # Connect the new output to insert the operator between
                        # the old output and the concat input
                        for i, x in enumerate(concat.inputs):
                            # Might actually be consumed multiple times...
                            if inp == x:
                                concat.replace_input_with(i, new)

                        # Insert the auxiliary operator into the list of
                        # producers to be moved
                        producers[index] = new.producer()

        # Insert all new initializers recorded by the tape
        for initializer in tape.initializers:
            # Register the initializer ir.Value recorded by the tape
            model.graph.register_initializer(initializer)
            # Mark the model as modified
            modified = True

        # Only modify the graph if new nodes are recorded for insertion
        if tape.nodes:
            # Insert collected nodes into the graph and ensure topological order
            model.graph.extend(tape.nodes)
            model.graph.sort()
            # Mark the model as modified
            modified = True

        # Potentially modified model and indicator whether the model actually
        # changed
        return ir.passes.PassResult(model, modified)
