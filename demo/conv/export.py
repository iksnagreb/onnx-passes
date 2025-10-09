# Creating and exporting the test model
import torch
# For saving reference input/output for verification
import numpy as np

if __name__ == "__main__":
    # Configurable convolution layer - change any options and reexport to
    # explore lowering
    model = torch.nn.LazyConv2d(
        # Number of channels to produce at the output - for depthwise match the
        # input channels and the groups below
        out_channels=6,
        # Configures the sliding window and padding of the feature map
        # TODO: Padding disabled to demonstrate layout converter...
        kernel_size=(3, 3), padding=(0, 0), stride=(1, 1), dilation=(1, 1),
        # Split convolutions into groups along the channel axis (both input and
        # output) - match with channels for depthwise
        groups=1,
        # Add a bias along the channel dimension as part of the operator - will
        # be lowered into a standalone Add operator
        bias=True
    )

    # Generate some random data in image-like layout for export and verification
    x = torch.rand(1, 1, 28, 28)
    y = model(x)
    torch.onnx.export(
        model, (x,), "model.onnx", dynamo=True, external_data=False,
        opset_version=19
    )

    # Save verification input/output pair
    np.save("inp.npy", x.detach().numpy())
    np.save("out.npy", y.detach().numpy())
