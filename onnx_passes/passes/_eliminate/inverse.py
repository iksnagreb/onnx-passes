from onnx_passes.passes._base import RewriteRule
from onnx_passes.passes._verify import Verify, tolerance

import onnx_ir as ir
import numpy as np


def match_constant(c):
    """Generate a value level checker for a constant value."""

    def check(_, value: ir.Value):
        if (value := ir.convenience.get_const_tensor(value)) is not None:
            return np.all(value.numpy() == c)
        return False

    return check


@tolerance
class EliminateInverseAdd_v1(RewriteRule, Verify):
    """Eliminates addition of the additive inverse element."""

    @staticmethod
    def pattern(op, x):
        return op.Add(x, op.Mul(match_constant(-1), x), _outputs=["out"])

    @staticmethod
    def rewrite(op, x, out):
        return op.Expand(
            op.CastLike(
                op.Constant(value_int=0),
                x
            ),
            # Ensure valid shape according to broadcasting
            op.Shape(
                op.Add(
                    x,
                    op.Mul(
                        op.ConstantOfShape(
                            op.Shape(out.producer().inputs[0])
                        ),
                        op.ConstantOfShape(
                            op.Shape(out.producer().inputs[1])
                        )
                    )
                )
            )
        )


@tolerance
class EliminateInverseMul_v1(RewriteRule, Verify):
    """Eliminates multiplication of the multiplicative inverse element."""

    @staticmethod
    def pattern(op, x):
        return op.Mul(x, op.Reciprocal(x))

    @staticmethod
    def rewrite(op, x):
        return op.Expand(
            op.CastLike(
                op.Constant(value_int=1),
                x
            ),
            op.Shape(
                x
            )
        )
