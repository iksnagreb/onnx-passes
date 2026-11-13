from onnx_passes.ops import link_ops

from onnx.reference import ReferenceEvaluator

import onnx_ir as ir
import numpy as np


def batched_inputs(model: ir.Model, inputs: list) -> list[list] | None:
    """Cut inputs into batches matching the models batch size, i.e., first axis.

    If the size of inputs is not divisible by the batch size, the last batch is
    shorter. Either pad or drop the shorter batch for correct model evaluation.
    """

    # Figure out the common batch size of all inputs and return None if there is
    # no unique common size to batch all inputs
    batch_size: int | None = None

    for tensor, _ in zip(model.graph.inputs, inputs):
        if (shape := tensor.shape) is not None and shape.is_static():
            if batch_size is None:
                batch_size = shape[0]  # noqa: shape is not symbolic

            if shape[0] != batch_size:
                return None
        else:
            return None

    # Split all inputs into batches of the common batch size along the first
    # axis. The final batch might be smaller and should be padded for eval.
    batches = []

    for value in inputs:
        batches.append(
            np.split(
                value,
                list(
                    range(batch_size, len(value), batch_size)  # noqa: not None
                )
            )
        )

    # Transpose from inputs x batches to batches x inputs
    return list(map(list, zip(*batches)))


def evaluate_model(model: ir.Model, inputs: list):
    """Evaluate larger inputs cut into batches matching the model input size."""

    # Figure out the common batch size of all inputs and return None if there is
    # no unique common size to batch all inputs
    batch_size: int | None = None

    for tensor, _ in zip(model.graph.inputs, inputs):
        if (shape := tensor.shape) is not None and shape.is_static():
            if batch_size is None:
                batch_size = shape[0]  # noqa: shape is not symbolic

            if shape[0] != batch_size:
                batch_size = None
        else:
            batch_size = None

    # Link all custom operator functions into the mode graph and serialize to
    # proto representation as used by the reference evaluator
    proto = ir.to_proto(link_ops(model))

    outputs, contexts = None, []

    # Batched evaluation if a common batch size has been found to cut all inputs
    # into chunks of the same size along the first axis.
    if batch_size and (batches := batched_inputs(model, inputs)) is not None:
        for batch in batches:
            # Resize all inputs to match the batch size, assuming no input is
            # longer than the batch size
            size = None

            for i, value in enumerate(batch):
                if size is None:
                    size = value.shape[0]

                batch[i] = np.resize(value, (batch_size, *value.shape[1:]))

            # Fill the execution context with the input values assuming they
            # correspond to the model graph inputs in correct order
            context = {}

            for tensor, value in zip(model.graph.inputs, batch):
                if (dtype := tensor.dtype) is not None:
                    value = np.asarray(value, dtype.numpy())
                context[tensor.name] = np.asarray(value)

            # Use the ONNX reference evaluator to execute the model on the
            # context collecting named outputs into the context as well
            session = ReferenceEvaluator(proto)
            context = session.run(None, context, intermediate=True)

            # Collect the outputs from the execution context assuming they
            # correspond to the model graph outputs in correct order
            output = []

            for tensor in model.graph.outputs:
                output.append(np.asarray(context[tensor.name]))

            for i, value in enumerate(output):
                output[i] = value[:size]

            if outputs is not None:
                for i, value in enumerate(output):
                    outputs[i] = np.concat((outputs[i], value), axis=0)
            else:
                outputs = output

            contexts.append(context)

        if len(contexts) == 1:
            contexts = contexts[0]
    # Non-batched evaluation if no common batch size has been found. Best effort
    # evaluation, assuming all inputs match the corresponding model input.
    else:
        # Fill the execution context with the input values assuming they
        # correspond to the model graph inputs in correct order
        context = {}

        for tensor, value in zip(model.graph.inputs, inputs):
            if (dtype := tensor.dtype) is not None:
                value = np.asarray(value, dtype.numpy())
            context[tensor.name] = np.asarray(value)

        # Use the ONNX reference evaluator to execute the model on the
        # context collecting named outputs into the context as well
        session = ReferenceEvaluator(proto)
        context = session.run(None, context, intermediate=True)

        # Collect the outputs from the execution context assuming they
        # correspond to the model graph outputs in correct order
        outputs = []

        for tensor in model.graph.outputs:
            outputs.append(np.asarray(context[tensor.name]))

        contexts = context

    return outputs, contexts
