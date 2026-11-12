from onnx_passes.passes.base import RewriteRule
from onnx_passes.passes.verify import Verify


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
    """Remove permutations of singleton dimensions.

    Singleton axes can be assigned to the output positions receiving singleton
    axes in any order without changing the result of the transpose. Assigning
    them in ascending order reverts permutations among singleton axes while
    leaving the permutation of all other axes untouched.
    """

    # Output positions receiving a singleton axis of the input
    positions = [i for i, axis in enumerate(perm) if shape[axis] == 1]

    # Reassign the singleton axes to these positions in ascending order
    perm = list(perm)

    for i, axis in zip(positions, sorted(perm[i] for i in positions)):
        perm[i] = axis

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
