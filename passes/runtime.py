# ir.Model, ir.from_proto, ir.to_proto, ...
import onnx_ir as ir

# The runtime simply builds a wrapper around ONNX Runtime for model execution
import onnxruntime


# Evaluates the model on the inputs via ONNX Runtime inference session
def evaluate_model(model: ir.Model, inputs: list, **kwargs):
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

    # Convert the model to a string-serialized protobuf representation
    # understood by ONNX Runtime
    model = ir.to_proto(model).SerializeToString()
    # Create an inference session from the ONNX model converted to proto
    # representation
    session = onnxruntime.InferenceSession(model, **kwargs)

    # Fill the execution context with inputs paired-up with the corresponding
    # input names from the model graph
    # TODO: Check if some mechanism is necessary to ensure input order is
    #  preserved through all of the flow...
    context = {
        inp.name: x for inp, x in zip(session.get_inputs(), inputs)
    }

    # Evaluate the model on the inputs form the execution context by running the
    # prepared inference session and collect all outputs as results
    return session.run(None, context)
