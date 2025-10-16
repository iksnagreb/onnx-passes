# Op types of ONNX operators categorized as a pooling operation, these can be
# interpreted as an input generator followed by a reduction operation
POOLING = {
    "AveragePool",
    "GlobalAveragePool",
    "GlobalLpPool",
    "GlobalMaxPool",
    "LpPool",
    "MaxPool",
}

# ONNX IR operator node representation
from onnx_ir import Node


def is_pooling(op: str | Node):
    if isinstance(op, Node):
        return op.op_type in POOLING
    return op in POOLING
