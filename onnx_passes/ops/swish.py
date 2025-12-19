# Custom operator function registry
from onnx_passes.ops import register, op


# Add a custom Swish function to the domain as standard ONNX provides Swish only
# since opset 24.
@register
def Swish(x, alpha: float = 1.0):
    return x * op.Sigmoid(alpha * x)


# Define Silu function in terms of Swish (this is pretty much just an alias to
# the default Swish alpha=1).
@register
def Silu(x):
    return Swish(x, alpha=1.0)
