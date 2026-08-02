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


def _remove_singleton_transpose(perm, shape: tuple[int, ...]):
    """Remove permutations of singleton dimensions."""

    # Find permutations of singleton dimensions
    singleton_perm = {}

    for axis in perm:
        if shape[axis] == shape[perm[axis]] == 1:
            if axis != perm[axis]:
                singleton_perm[axis] = perm[axis]

    # Revert permutations of singleton dimensions
    perm = list(perm)

    for i, axis in enumerate(perm):
        if axis in singleton_perm:
            perm[i] = singleton_perm[axis]

    return tuple(perm)


class RemoveSingletonTranspose_v1(RewriteRule, Verify):
    """Removes permutations of singleton axes from transpose."""

    @staticmethod
    def pattern(op, x, perm):
        return op.Transpose(x, perm=perm)

    @staticmethod
    def check(op, x, perm):
        if x.shape is not None and x.shape.is_static():
            perm = perm.as_ints()
            return _remove_singleton_transpose(perm, x.shape) != perm
        return False

    @staticmethod
    def rewrite(op, x, perm):
        return op.Transpose(
            x, perm=_remove_singleton_transpose(perm.as_ints(), x.shape)
        )
