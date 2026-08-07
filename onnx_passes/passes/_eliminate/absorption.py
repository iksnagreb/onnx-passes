from onnx_passes.passes._base import RewriteRuleSetTemplate
from onnx_passes.passes._verify import Verify


class EliminateAbsorption_v1(RewriteRuleSetTemplate, Verify):
    """Eliminate absorption identity of boolean algebras and lattices."""

    patterns = (
        lambda op: (op.And, op.Or),
        lambda op: (op.Or, op.And),
        lambda op: (op.Min, op.Max),
        lambda op: (op.Max, op.Min),
    )

    # Note: So far all these operations are commutative, thus one instance per
    # operation pair is sufficient. Once adding non-commutative operations, up
    # to four different orders need to be spelled out:
    #   x . (x * y) -> x    (1)
    #   x . (y * x) -> x    (2)
    #   (x * y) . x -> x    (3)
    #   (y * x) . x -> x    (4)

    @staticmethod
    def pattern(partial, op, x, y):
        return partial(op)[0](x, partial(op)[1](x, y))

    @staticmethod
    def rewrite(partial, op, x, y):
        return op.Expand(x, op.Shape(partial(op)[1](x, y)))
