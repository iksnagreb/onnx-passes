# ONNX IR for saving models from protobuf representation
import onnx_ir as ir

# For generating and saving test data
import numpy as np

# Build the demo model using ONNX Script opset version 15
from onnxscript import opset15 as op
from onnxscript import FLOAT, script

# Configurable dimensions of the demo model: Batch, Channel-in, Channel-out
N, C, D = 32, 256, 32
# Generate random model weight parameter
W = np.random.rand(C, D).astype(np.float32)


# Demo model applying MatMul and reshaping to some inputs
@script(opset=op, default_opset=op)
def model(X: FLOAT[N, C]) -> FLOAT[N, D, 1]:  # noqa
    # Some test pattern of multiplications and additions around MatMul
    y = (0.1 * (2.3 * (1.2 * X))) @ op.Constant(value=W) + 0.5 + 0.2  # noqa
    # Derive the output shape by expanding an axis
    shape = op.Concat(op.Shape(y), [1], axis=-1)
    # Apply some dynamically calculated shape
    return op.Reshape(y, shape=shape)


# Generate some inputs and produce output via eager mode evaluation
x = np.random.rand(N, C).astype(np.float32)
y = model(x)

# Save input and output data for model verification
np.save("inp.npy", x)
# Introduce a small deviation from the true output: Setting the range of
# accepted absolute error for verification in cfg.yaml:verify below this should
# demonstrate a VerificationError
np.save("out.npy", y + 0.001)

# Save the model to ONNX file
ir.save(ir.from_proto(model.to_model_proto()), "model.onnx")
