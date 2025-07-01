# ir.Model, ir.passes.PassResult, ir.from_proto, ir.to_proto, ...
import onnx_ir as ir

# Recursively iterate nodes from model graphs and function in order
from onnx_ir.traversal import RecursiveGraphIterator

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes


# Gives unique names to each node by enumerating nodes per operator type
@passes.verify.equality
@passes.register("cleanup")
@passes.register("unique-names")
class GiveUniqueNodeNames(passes.base.Transformation):
    # Applies the transformation to the model graph given as ONNX IR
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Track whether any node actually changed
        modified = False
        # Dictionary tracking the per operator type count
        counts = {}
        # Modify a deep copy of the original model
        model = ir.from_proto(ir.to_proto(model))

        # Iterate all nodes in the graph
        for node in RecursiveGraphIterator(model.graph):
            # Look up the current per operator type count
            count = counts.setdefault(node.op_type, 0)
            # Derive a name reflecting the operator type and count
            name = f"{node.op_type}_{count}"
            # Check whether the node already has this name and consider the
            # model to have changed if at least one name changes
            modified = modified or name != node.name
            # Assign the new name after checking for change
            node.name = name
            # Increment the per operator type count
            counts[node.op_type] += 1

        # We prefer functional passes - return a deep copy to be modified
        return ir.passes.PassResult(model, modified)


# Gives readable names to each tensor by deriving the name from the producer
# node - or the first consumer node if initializers or globals if graph inputs
# or outputs.
@passes.verify.equality
@passes.register("cleanup")
@passes.register("unique-names")
class GiveReadableTensorNames(passes.base.Transformation):
    # Applies the transformation to the model graph given as ONNX IR
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Track whether any node actually changed
        modified = False
        # Modify a deep copy of the original model
        model = ir.from_proto(ir.to_proto(model))
        # Start counting global inputs and outputs to the graph
        inputs, outputs = [], []

        # Iterate all nodes in the graph - reverse iteration to not overwrite
        # output names by following input names
        for node in RecursiveGraphIterator(model.graph, reverse=True):
            # Enumerate all inputs to the node
            for i, inp in enumerate(node.inputs):
                # Optional inputs can be represented by None...
                if inp is not None:
                    # Select a different name to reflect global inputs
                    if inp.is_graph_input() and not inp.is_initializer():
                        # Collect global inputs to update the name later
                        inputs.append(inp)
                    # Derive a new name by enumerating all inputs to the node
                    name = f"{node.name}_input_{i}"
                    # Check whether the value already has this name and consider
                    # the model to have changed if at least one name changes
                    modified = modified or name != node.inputs[i].name  # noqa
                    # Assign the new name after checking for change
                    # TODO: According to the specification, inputs should be
                    #  immutable, however, this seems to work fine...
                    node.inputs[i].name = name

            # Enumerate all outputs from the node
            for i, out in enumerate(node.outputs):
                # Optional outputs can be represented by None...
                if out is not None:
                    # Select a different name to reflect global outputs
                    if out.is_graph_output() and not out.is_initializer():
                        # Collect global outputs to update the name later
                        outputs.append(out)
                    # Derive a new name by enumerating all outputs to the node
                    name = f"{node.name}_output_{i}"
                    # Check whether the value already has this name and consider
                    # the model to have changed if at least one name changes
                    modified = modified or name != node.outputs[i].name  # noqa
                    # Assign the new name after checking for change
                    # TODO: According to the specification, outputs should be
                    #  immutable, however, this seems to work fine...
                    node.outputs[i].name = name

        # Enumerate all inputs to the model graph to assign new names
        for i, inp in enumerate(set(inputs)):
            # Derive a new name by enumerating all inputs to the model
            name = f"global_input_{i}"
            # Check whether the value already has this name and consider the
            # model to have changed if at least one name changes
            modified = modified or name != inp.name  # noqa: []?
            # Assign the new name after checking for change
            # TODO: According to the specification, inputs should be
            #  immutable, however, this seems to work fine..
            inp.name = name

        # Enumerate all outputs to the model graph to assign new names
        for i, out in enumerate(set(outputs)):
            # Derive a new name by enumerating all outputs to the model
            name = f"global_output_{i}"
            # Check whether the value already has this name and consider the
            # model to have changed if at least one name changes
            modified = modified or name != out.name  # noqa: []?
            # Assign the new name after checking for change
            # TODO: According to the specification, outputs should be
            #  immutable, however, this seems to work fine..
            out.name = name

        # We prefer functional passes - return a deep copy to be modified
        # TODO: Seems to never stop when returning the true modification state
        #  as iterations overwrite the result of previous over and over again...
        return ir.passes.PassResult(model, False)
