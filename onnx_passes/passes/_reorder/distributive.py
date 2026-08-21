from onnx_passes.passes._base import (
    RewriteRuleSetTemplate, Transformation, Sequential
)
from onnx_passes.passes._verify import Verify, tolerance

from onnxscript.rewriter.pattern import OrValue

import onnx_ir as ir

_DISTRIBUTIVE_PATTERNS = (
    lambda op: (op.Mul, op.Add),
    lambda op: (op.Max, op.Min),
    # lambda op: (op.Min, op.Max),
    lambda op: (op.And, op.Or),
    # lambda op: (op.Or, op.And),
    lambda op: (op.And, op.Xor),
    lambda op: (op.BitwiseAnd, op.BitwiseOr),
    # lambda op: (op.BitwiseOr, op.BitwiseAnd),
    lambda op: (op.BitwiseAnd, op.BitwiseXor),
    # Note: Even though matrix multiplication is distributive over addition,
    # MatMul broadcasting is not. Thus, it is not included here.
)


@tolerance
class ReorderDistributiveLhs_v1(RewriteRuleSetTemplate, Verify):
    """Reorder operations according to left distributivity."""

    patterns = _DISTRIBUTIVE_PATTERNS

    @staticmethod
    def pattern(partial, op, x, y, z):
        # Note: Refer to the two operations as multiplication and addition, just
        # as "multiplication distributed over addition", but these two could be
        # any distributive operations.
        mul, add = partial(op)
        return mul(x, add(y, z))

    @staticmethod
    def check(op, x, y, z):
        # Reorder configurations which bring together constants, allowing for
        # constant propagation without increasing the number of operations:
        #   c (y + c) -> cy + cc    constant foldable on the right
        #   c (c + z) -> cc + cz    constant foldable on the left
        x = ir.convenience.get_const_tensor(x) is not None
        y = ir.convenience.get_const_tensor(y) is not None
        z = ir.convenience.get_const_tensor(z) is not None

        if (x, y, z) in {(True, False, True), (True, True, False)}:
            return True

        return False

    @staticmethod
    def rewrite(partial, op, x, y, z):
        mul, add = partial(op)
        return add(mul(x, y), mul(x, z))


@tolerance
class ReorderReverseDistributiveLhs_v1(RewriteRuleSetTemplate, Verify):
    """Reorder operations according to reversed left distributivity."""

    patterns = _DISTRIBUTIVE_PATTERNS

    @staticmethod
    def pattern(partial, op, x, y, z):
        # Note: Refer to the two operations as multiplication and addition, just
        # as "multiplication distributed over addition", but these two could be
        # any distributive operations.
        mul, add = partial(op)
        return add(OrValue([mul(x, y), x]), OrValue([mul(x, z), x]))

    @staticmethod
    def check(context, x, y, z):
        # Reorder configurations pulling out a common term or bringing constants
        # together, decreasing the number of operations:
        #   xy + xz -> x (y + z)    3 -> 2 operations
        #   cy + cz -> c (y + z)    3 -> 2 operations, constant left
        #   xy + xc -> x (y + c)    3 -> 2 operations
        #   xc + xz -> x (c + z)    3 -> 2 operations
        #   xc + xc -> x (c + c)    3 -> 1 operations, constant foldable right
        # Note: This is the reverse of ReorderDistributiveLhs_v1

        if y is None or z is None:
            return True

        return not ReorderDistributiveLhs_v1.check(context, x, y, z)

    @staticmethod
    def rewrite(partial, op, x, y, z):
        mul, add = partial(op)

        if y is None:
            y = op.CastLike(op.Constant(value_int=1), x)

        if z is None:
            z = op.CastLike(op.Constant(value_int=1), x)

        return mul(x, add(y, z))


@tolerance
class ReorderDistributiveRhs_v1(RewriteRuleSetTemplate, Verify):
    """Reorder operations according to right distributivity."""

    patterns = _DISTRIBUTIVE_PATTERNS

    @staticmethod
    def pattern(partial, op, x, y, z):
        # Note: Refer to the two operations as multiplication and addition, just
        # as "multiplication distributed over addition", but these two could be
        # any distributive operations.
        mul, add = partial(op)
        return mul(add(x, y), z)

    @staticmethod
    def check(op, x, y, z):
        # Reorder configurations which bring together constants, allowing for
        # constant propagation without increasing the number of operations:
        #   (x + c) c -> xc + cc    constant foldable on the right
        #   (c + y) c -> cc + yc    constant foldable on the left
        x = ir.convenience.get_const_tensor(x) is not None
        y = ir.convenience.get_const_tensor(y) is not None
        z = ir.convenience.get_const_tensor(z) is not None

        if (x, y, z) in {(False, True, True), (True, False, True)}:
            return True

        return False

    @staticmethod
    def rewrite(partial, op, x, y, z):
        mul, add = partial(op)
        return add(mul(x, z), mul(y, z))


@tolerance
class ReorderReverseDistributiveRhs_v1(RewriteRuleSetTemplate, Verify):
    """Reorder operations according to reversed right distributivity."""

    patterns = _DISTRIBUTIVE_PATTERNS

    @staticmethod
    def pattern(partial, op, x, y, z):
        # Note: Refer to the two operations as multiplication and addition, just
        # as "multiplication distributed over addition", but these two could be
        # any distributive operations.
        mul, add = partial(op)
        return add(OrValue([mul(x, z), z]), OrValue([mul(y, z), z]))

    @staticmethod
    def check(context, x, y, z):
        # Reorder configurations pulling out a common term or bringing constants
        # together, decreasing the number of operations:
        #   xz + yz -> (x + y) z    3 -> 2 operations
        #   xc + yc -> (x + y) c    3 -> 2 operations, constant right
        #   xz + cz -> (x + c) z    3 -> 2 operations
        #   cz + yz -> (c + y) z    3 -> 2 operations
        #   cz + cz -> (c + c) z    3 -> 1 operations, constant foldable left
        # Note: This is the reverse of ReorderDistributiveRhs_v1

        if x is None or y is None:
            return True

        return not ReorderDistributiveRhs_v1.check(context, x, y, z)

    @staticmethod
    def rewrite(partial, op, x, y, z):
        mul, add = partial(op)

        if x is None:
            x = op.CastLike(op.Constant(value_int=1), z)

        if y is None:
            y = op.CastLike(op.Constant(value_int=1), z)

        return mul(add(x, y), z)


class ReorderDistributiveLoop_v1(Sequential, Transformation):
    """Exhaustively apply distributive reordering transformations."""

    passes = [
        ReorderDistributiveLhs_v1,
        ReorderReverseDistributiveLhs_v1,
        ReorderDistributiveRhs_v1,
        ReorderReverseDistributiveRhs_v1
    ]

    exhaustive = True
