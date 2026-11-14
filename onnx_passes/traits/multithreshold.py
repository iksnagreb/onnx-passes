import onnx_ir as ir


def is_multithreshold(op: str | ir.Node):
    if isinstance(op, ir.Node):
        return op.op_type == "MultiThreshold"
    return op == "MultiThreshold"


def produced_by_multithreshold(_, value: ir.Value) -> bool:
    """Check whether value is produced by a MultiThreshold operation."""
    return (node := value.producer()) is not None and is_multithreshold(node)
