import onnx_ir as ir
import numpy as np

from onnxscript import script, FLOAT, opset19 as op

# Draw random (normal distribution) weights in 32-bit floating-point
weights = np.random.randn(16, 32).astype(np.float32)  # noqa: this is ndarray

# Quantization bitwidths for inputs and weights
bits_x = 4
bits_w = 4

# Minimum and maximum quantized input, signed integer
min_x = 0
max_x = 2 ** bits_x - 1

# Minimum and maximum quantized weight, signed narrow range integer
min_w = -2 ** (bits_w - 1) + 1
max_w = +2 ** (bits_w - 1) - 1

# PowerQuant power parameter - in practice this would be determined via
# optimization over the weights of the entire model
alpha = 0.55

# Input quantization scale and bias assuming inputs from -1 to +1 shifted to 0
# to +2 to avoid issues with power of negative inputs
scale_x = np.float32(2.0 ** alpha) / np.float32(2 ** (bits_x - 1))
bias_x = -1

# PowerQuant weight quantization scale according to the paper
scale_w = np.max(np.abs(weights) ** alpha) / (2 ** (bits_w - 1) - 1)


@script(default_opset=op)
def model(x: FLOAT[1, 16]) -> FLOAT[1, 32]:
    # Matrix multiplication in PowerQuant quantized-dequantized format
    return op.MatMul(
        # PowerQuant input quantization and dequantization
        op.Add(
            op.Pow(
                op.Mul(
                    op.Clip(
                        op.Round(
                            op.Div(
                                op.Pow(
                                    op.Sub(
                                        x,
                                        op.Constant(value_float=bias_x)
                                    ),
                                    op.Constant(value_float=alpha)
                                ),
                                op.Constant(value_float=scale_x)
                            )
                        ),
                        op.Constant(value_float=min_x),
                        op.Constant(value_float=max_x),
                    ),
                    op.Constant(value_float=scale_x)
                ),
                op.Constant(value_float=1 / alpha)
            ),
            op.Constant(value_float=bias_x)
        ),
        # PowerQuant weight quantization and dequantization
        op.Mul(
            op.Sign(weights),
            op.Pow(
                op.Mul(
                    op.Abs(
                        op.Clip(
                            op.Round(
                                op.Div(
                                    op.Mul(
                                        op.Sign(weights),
                                        op.Pow(
                                            op.Abs(op.Constant(value=weights)),
                                            op.Constant(value_float=alpha)
                                        )
                                    ),
                                    op.Constant(value_float=scale_w)
                                )
                            ),
                            op.Constant(value_float=min_w),
                            op.Constant(value_float=max_w),
                        )
                    ),
                    op.Constant(value_float=scale_w)
                ),
                op.Constant(value_float=1 / alpha)
            )
        )
    )


if __name__ == "__main__":
    # Draw random (uniform) inputs between -1 and +1
    x = (2 * np.random.rand(1, 16) - 1).astype(np.float32)  # noqa: astype
    # Eager mode evaluation of the model in python to generate reference inputs
    # and outputs for verification
    np.save("inp.npy", x)
    np.save("out.npy", model(x))

    # Export the model tro ONNX proto representation
    model = ir.from_proto(model.to_model_proto())
    ir.save(model, "model.onnx")  # noqa: ir.Model
