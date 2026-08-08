from onnx_passes.passes._base import RewriteRule, RewriteRuleSet
from onnx_passes.passes._verify import Verify


class RewriteWhereAsMin_v1(RewriteRuleSet, Verify):
    """Rewrite conditional as minimum, i.e., x > y ? y : x = min(x, y)."""

    @staticmethod
    def pattern():
        return [  # noqa: Duplicate...
            lambda op, x, y: op.Where(op.Greater(x, y), y, x),
            lambda op, x, y: op.Where(op.GreaterOrEqual(x, y), y, x),
            lambda op, x, y: op.Where(op.Less(x, y), x, y),
            lambda op, x, y: op.Where(op.LessOrEqual(x, y), x, y),
        ]

    @staticmethod
    def rewrite():
        return [
            lambda op, x, y: op.Min(x, y),
            lambda op, x, y: op.Min(x, y),
            lambda op, x, y: op.Min(x, y),
            lambda op, x, y: op.Min(x, y),
        ]


class RewriteWhereAsMax_v1(RewriteRuleSet, Verify):
    """Rewrite conditional as maximum, i.e., x > y ? x : y = max(x, y)."""

    @staticmethod
    def pattern():
        return [  # noqa: Duplicate...
            lambda op, x, y: op.Where(op.Greater(x, y), x, y),
            lambda op, x, y: op.Where(op.GreaterOrEqual(x, y), x, y),
            lambda op, x, y: op.Where(op.Less(x, y), y, x),
            lambda op, x, y: op.Where(op.LessOrEqual(x, y), y, x),
        ]

    @staticmethod
    def rewrite():
        return [
            lambda op, x, y: op.Max(x, y),
            lambda op, x, y: op.Max(x, y),
            lambda op, x, y: op.Max(x, y),
            lambda op, x, y: op.Max(x, y),
        ]


class RewriteClipAsMinMax_v1(RewriteRule, Verify):
    """Rewrite clipping and minimum-maximum combination."""

    @staticmethod
    def pattern(op, x, minimum, maximum):
        return op.Clip(x, minimum, maximum)

    @staticmethod
    def rewrite(op, x, minimum, maximum):
        return op.Min(op.Max(x, minimum), maximum)


class UnrollMin_v1(RewriteRule, Verify):
    """Rewrite multi-input Min as a tree/chain of binary Min operations."""

    @staticmethod
    def pattern(op, x, y, z):
        return op.Min(x, y, z, _allow_other_inputs=True, _outputs=["out"])

    @staticmethod
    def rewrite(op, x, y, z, out):
        x = op.Min(op.Min(x, y), z)

        for value in out.producer().inputs[3:]:
            x = op.Min(x, value)

        return x


class UnrollMax_v1(RewriteRule, Verify):
    """Rewrite multi-input Max as a tree/chain of binary Max operations."""

    @staticmethod
    def pattern(op, x, y, z):
        return op.Max(x, y, z, _allow_other_inputs=True, _outputs=["out"])

    @staticmethod
    def rewrite(op, x, y, z, out):
        x = op.Max(op.Max(x, y), z)

        for value in out.producer().inputs[3:]:
            x = op.Max(x, value)

        return x
