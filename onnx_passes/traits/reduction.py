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

import onnx_ir as ir


def is_reduction(op: str | ir.Node):
    if isinstance(op, ir.Node):
        return op.op_type in REDUCTIONS
    return op in REDUCTIONS


def produced_by_reduction(_, value: ir.Value) -> bool:
    """Check whether value is produced by a reduction operation."""
    return (node := value.producer()) is not None and is_reduction(node)
