from onnx_passes.passes._base import RewriteRule
from onnx_passes.passes._verify import Verify


class InferTransposePerm_v1(RewriteRule, Verify):
    """Infers transpose permutation if no attribute is given."""

    @staticmethod
    def pattern(op, x):
        return op.Transpose(x, _outputs=["y"])

    @staticmethod
    def check(op, x, y):
        if (_ := y.producer().attributes.get("perm", None)) is None:
            return x.shape is not None and x.shape.is_static()
        return False

    @staticmethod
    def rewrite(op, x, y):
        return op.Transpose(x, perm=list(reversed(range(len(x.shape)))))


class FuseConsecutiveTransposes_v1(RewriteRule, Verify):
    """Fuses two consecutive transpose operations into a single transpose."""

    @staticmethod
    def pattern(op, x, perm1, perm2):
        return op.Transpose(op.Transpose(x, perm=perm1), perm=perm2)

    @staticmethod
    def rewrite(op, x, perm1, perm2):
        return op.Transpose(
            x, perm=[perm1.as_ints()[i] for i in perm2.as_ints()]
        )
