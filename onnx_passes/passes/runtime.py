# ir.Model, ir.from_proto, ir.to_proto, ...
import onnx_ir as ir

# The runtime simply builds a wrapper around ONNX Runtime for model execution
import onnxruntime, onnxruntime_extensions


# Evaluates the model on the inputs via ONNX Runtime inference session
def evaluate_model(model: ir.Model, inputs: list,
                   full_context_dump: bool = False, **kwargs):
    # Sanitize the providers field if present - must be either just a list of
    # strings or a list of tuples of string and dict
    def _sanitize(provider):
        # Strings are augmented by an empty parameter dictionary
        if isinstance(provider, str):
            return provider, {}
        # Probably a list of provider name and optional arguments
        provider, *args = provider
        # Insert empty dictionary if no arguments are provided
        return provider, dict(*(args if args else {}))

    # Make sure we always have a provider field
    kwargs.setdefault("providers", [])

    # If at least one provider has arguments specified, sanitize them all to
    # have empty arguments
    if not all(isinstance(provider, str) for provider in kwargs["providers"]):
        kwargs["providers"] = [_sanitize(args) for args in kwargs["providers"]]

    # Load DLLs to make the CUDA execution provider available
    onnxruntime.preload_dlls()

    # Make a deep copy of the model to not mess up the graph by executing it...
    model = ir.from_proto(ir.to_proto(model))
    # Remember the original list of outputs before extending by all
    # intermediate tensors
    outputs = list(model.graph.outputs)

    # Start collecting a list of intermediate tensor value information
    intermediates = []

    # Optionally extends the list of model outputs by all intermediate tensors,
    # i.e., produces a full execution context dump
    if full_context_dump:
        # Collect all tensors which are actually used in the graph, i.e.,
        # connected to any node as either input or output
        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            intermediates.extend([*node.inputs, *node.outputs])

    # Filter out all tensors which are already graph outputs and remove all
    # duplicates by turning the list into a set
    intermediates = {x for x in intermediates if not x.is_graph_output()}
    # Extend the list of graph outputs by all these intermediate (or input)
    # tensors - this keeps the original outputs first
    model.graph.outputs.extend(intermediates)

    # Convert the model to a string-serialized protobuf representation
    # understood by ONNX Runtime
    model = ir.to_proto(model).SerializeToString()

    # Disable further ONNX Runtime session graph optimizations
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    # Only show error and fatal messages
    sess_options.log_severity_level = 3

    # Path to custom operators provided by ONNX Runtime Extensions
    extensions_library_path = onnxruntime_extensions.get_library_path()
    # Register the extension library to make custom operators available
    sess_options.register_custom_ops_library(extensions_library_path)

    # Create an inference session from the ONNX model converted to proto
    # representation
    session = onnxruntime.InferenceSession(model, sess_options, **kwargs)

    # Fill the execution context with inputs paired-up with the corresponding
    # input names from the model graph
    # TODO: Check if some mechanism is necessary to ensure input order is
    #  preserved through all of the flow...
    context = {
        inp.name: x for inp, x in zip(session.get_inputs(), inputs)
    }

    # Evaluate the model on the inputs form the execution context by running the
    # prepared inference session and collect all outputs as results
    results = session.run(None, context)

    # Collect full execution context by associating each output, including the
    # intermediates, to its name
    context = {
        **{out.name: x for x, out in zip(results[:len(outputs)], outputs)},
        **{out.name: x for x, out in zip(results[len(outputs):], intermediates)}
    }

    # Collect only the original outputs as list to match original model
    # execution behavior of "outputs = f(inputs)"
    outputs = [x for x, _ in zip(results, outputs)]

    # Return both, the original outputs as list and the full execution contex as
    # a dictionary
    return outputs, context
