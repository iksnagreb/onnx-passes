from onnx_passes.passes._base import RewriteRuleSetTemplate
from onnx_passes.passes._verify import Verify

import onnx_ir as ir


def match_constant(_, value: ir.Value) -> bool:
    """Value level checker for constant values."""
    return ir.convenience.get_const_tensor(value) is not None


class ReorderCommutative_v1(RewriteRuleSetTemplate, Verify):
    """Reorder commutative operations to move constants to the right."""

    patterns = (
        lambda op: op.Add,
        lambda op: op.Mul,
        lambda op: op.Max,
        lambda op: op.Min,
        lambda op: op.Or,
        lambda op: op.And,
        lambda op: op.Xor,
        lambda op: op.BitwiseOr,
        lambda op: op.BitwiseAnd,
        lambda op: op.BitwiseXor,
    )

    @staticmethod
    def pattern(partial, op, y):
        return partial(op)(match_constant, y, _outputs=["out"])

    @staticmethod
    def check(context, y, out):
        return not match_constant(context, y)

    @staticmethod
    def rewrite(partial, op, y, out):
        return partial(op)(y, out.producer().inputs[0])
