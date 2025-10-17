# Op types of ONNX operators categorized as a reduction operation, many of these
# are actually special cases of ReduceSum and are implemented as ONNX functions
REDUCTIONS = [
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceMax",
    "ReduceMean",
    "ReduceMin",
    "ReduceProd",
    "ReduceSum",
    "ReduceSumSquare",
]

# ONNX IR operator node representation
from onnx_ir import Node


def is_reduction(op: str | Node):
    if isinstance(op, Node):
        return op.op_type in REDUCTIONS
    return op in REDUCTIONS
