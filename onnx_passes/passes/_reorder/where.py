from onnx_passes.passes._base import RewriteRule
from onnx_passes.passes._verify import Verify

from onnx_passes.traits.elementwise import produced_by_elementwise


class MoveElementwisePastWhere_v1(RewriteRule, Verify):
    """Reorder common elementwise operations to follow Where."""

    @staticmethod
    def pattern(op, condition):
        return op.Where(condition, _allow_other_inputs=True, _outputs=["out"])

    @staticmethod
    def check(context, condition, out):
        if not produced_by_elementwise(
                context, lhs := out.producer().inputs[1]
        ):
            return False

        if not produced_by_elementwise(
                context, rhs := out.producer().inputs[2]
        ):
            return False

        if lhs.producer().op_type == rhs.producer().op_type:
            if lhs.producer().attributes == rhs.producer().attributes:
                return lhs.producer().inputs[1:] == rhs.producer().inputs[1:]

        return False

    @staticmethod
    def rewrite(op, condition, out):
        lhs, rhs = out.producer().inputs[1:]

        return op.op(
            lhs.producer().op_type,
            op.Where(
                condition,
                lhs.producer().inputs[0],
                rhs.producer().inputs[0]
            ),
            *lhs.producer().inputs[1:],
            **lhs.producer().attributes
        )
