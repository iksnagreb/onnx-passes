# ir.Model, ir.from_proto, ir.to_proto, ...
import onnx_ir as ir

# The runtime simpyl builds a wrapper around ONNX Runtime for model execution
import onnxruntime


# Evaluates the model on the inputs via ONNX Runtime inference session
def evaluate_model(model: ir.Model, inputs: list, **_):
    # TODO: Build session and provider options from keyword arguments to
    #  configure the inference session, e.g., to make use of GPU-acceleration

    # Convert the model to a string-serialized protobuf representation
    # understood by ONNX Runtime
    model = ir.to_proto(model).SerializeToString()
    # Create an inference session from the ONNX model converted to proto
    # representation
    session = onnxruntime.InferenceSession(model)

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
