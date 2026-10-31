import onnx_ir as ir


def produced_by_expand(_, value: ir.Value) -> bool:
    """Check whether value is produced by an Expand operation."""
    return (node := value.producer()) is not None and node.op_type == "Expand"
