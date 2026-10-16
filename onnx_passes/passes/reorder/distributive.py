from onnx_passes.passes.base import (
    RewriteRule, RewriteRuleSetTemplate, Sequential
)
from onnx_passes.passes.verify import Verify, tolerance

from onnxscript.rewriter.pattern import OrValue

import onnx_ir as ir
import numpy as np


@tolerance
class ReorderDistributiveLhs_v1(RewriteRuleSetTemplate, Verify):
    """Reorder operations according to left distributivity."""

    patterns = (
        lambda op: (op.Mul, op.Add),
        lambda op: (op.Max, op.Min),
        lambda op: (op.And, op.Or),
        lambda op: (op.And, op.Xor),
        lambda op: (op.BitwiseAnd, op.BitwiseOr),
        lambda op: (op.BitwiseAnd, op.BitwiseXor),
        # Note: Even though matrix multiplication is distributive over addition,
        # MatMul broadcasting is not. Thus, it is not included here.
    )

    @staticmethod
    def pattern(partial, op, x, y, z):
        # Note: Refer to the two operations as multiplication and addition, just
        # as "multiplication distributed over addition", but these two could be
        # any distributive operations.
        mul, add = partial(op)
        return mul(x, add(y, z))

    @staticmethod
    def check(context, x, y, z):
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

    # Pattern: (mul-like, add-like, identity), where identity is the identity
    # element to insert for implicit constants
    patterns = (
        lambda op: (op.Mul, op.Add, 1),
        lambda op: (op.Max, op.Min, -np.inf),
        # lambda op: (op.Min, op.Max),
        lambda op: (op.And, op.Or, True),
        # lambda op: (op.Or, op.And),
        lambda op: (op.And, op.Xor, True),
        lambda op: (op.BitwiseAnd, op.BitwiseOr, ~0),
        # lambda op: (op.BitwiseOr, op.BitwiseAnd),
        lambda op: (op.BitwiseAnd, op.BitwiseXor, ~0),
        # Note: Even though matrix multiplication is distributive over addition,
        # MatMul broadcasting is not. Thus, it is not included here.
    )

    @staticmethod
    def pattern(partial, op, x, y, z):
        # Note: Refer to the two operations as multiplication and addition, just
        # as "multiplication distributed over addition", but these two could be
        # any distributive operations.
        mul, add, identity = partial(op)
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

        if y is None and z is not None:
            # x + xz -> x (1 + z)
            return ir.convenience.get_const_tensor(z) is not None

        if z is None and y is not None:
            # xy + x -> x (y + 1)
            return ir.convenience.get_const_tensor(y) is not None

        if y is None and z is None:
            return True

        return not ReorderDistributiveLhs_v1.check(context, x, y, z)

    @staticmethod
    def rewrite(partial, op, x, y, z):
        mul, add, identity = partial(op)

        if y is None:
            y = op.CastLike(op.Constant(value=ir.tensor(identity)), x)

        if z is None:
            z = op.CastLike(op.Constant(value=ir.tensor(identity)), x)

        return mul(x, add(y, z))


@tolerance
class ReorderDistributiveRhs_v1(RewriteRuleSetTemplate, Verify):
    """Reorder operations according to right distributivity."""

    patterns = (
        lambda op: (op.Mul, op.Add),
        lambda op: (op.Max, op.Min),
        lambda op: (op.And, op.Or),
        lambda op: (op.And, op.Xor),
        lambda op: (op.BitwiseAnd, op.BitwiseOr),
        lambda op: (op.BitwiseAnd, op.BitwiseXor),
        # Note: Even though matrix multiplication is distributive over addition,
        # MatMul broadcasting is not. Thus, it is not included here.
    )

    @staticmethod
    def pattern(partial, op, x, y, z):
        # Note: Refer to the two operations as multiplication and addition, just
        # as "multiplication distributed over addition", but these two could be
        # any distributive operations.
        mul, add = partial(op)
        return mul(add(x, y), z)

    @staticmethod
    def check(context, x, y, z):
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

    # Pattern: (mul-like, add-like, identity), where identity is the identity
    # element to insert for implicit constants
    patterns = (
        lambda op: (op.Mul, op.Add, 1),
        lambda op: (op.Max, op.Min, -np.inf),
        # lambda op: (op.Min, op.Max),
        lambda op: (op.And, op.Or, True),
        # lambda op: (op.Or, op.And),
        lambda op: (op.And, op.Xor, True),
        lambda op: (op.BitwiseAnd, op.BitwiseOr, ~0),
        # lambda op: (op.BitwiseOr, op.BitwiseAnd),
        lambda op: (op.BitwiseAnd, op.BitwiseXor, ~0),
        # Note: Even though matrix multiplication is distributive over addition,
        # MatMul broadcasting is not. Thus, it is not included here.
    )

    @staticmethod
    def pattern(partial, op, x, y, z):
        # Note: Refer to the two operations as multiplication and addition, just
        # as "multiplication distributed over addition", but these two could be
        # any distributive operations.
        mul, add, identity = partial(op)
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

        if x is None and y is not None:
            # z + yz -> (1 + y) z
            return ir.convenience.get_const_tensor(y) is not None

        if y is None and x is not None:
            # xz + z -> (x + 1) z
            return ir.convenience.get_const_tensor(x) is not None

        if x is None and y is None:
            return True

        return not ReorderDistributiveRhs_v1.check(context, x, y, z)

    @staticmethod
    def rewrite(partial, op, x, y, z):
        mul, add, identity = partial(op)

        if x is None:
            x = op.CastLike(op.Constant(value=ir.tensor(identity)), z)

        if y is None:
            y = op.CastLike(op.Constant(value=ir.tensor(identity)), z)

        return mul(add(x, y), z)


@tolerance
class ReorderMatMulDistributiveLhs_v1(RewriteRule, Verify):
    """Reorder MatMul according to left distributivity."""

    @staticmethod
    def pattern(op, x, y, z):
        return op.MatMul(x, op.Add(y, z))

    @staticmethod
    def check(context, x, y, z):
        # Dynamic or missing shapes would lead to endless back and forth between
        # forward and reverse distributive rules (if we do not force constant
        # folding as part of every cleanup)
        if y.shape is None or y.shape.is_dynamic():
            if z.shape is None or z.shape.is_dynamic():
                return False

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
    def rewrite(op, x, y, z):
        # Avoid endless back and forth between forward and reverse distributive
        # rules by statically expanding and computing the constant side.
        if (value := ir.convenience.get_const_tensor(y)) is not None:
            return op.Add(
                op.Constant(
                    value=ir.tensor(
                        np.matmul(
                            ir.convenience.get_const_tensor(x),
                            np.broadcast_to(
                                value.numpy(),
                                common_shape := np.broadcast_shapes(
                                    y.shape, z.shape
                                )
                            )
                        )
                    )
                ),
                op.MatMul(
                    x,
                    op.Expand(
                        z,
                        op.Constant(
                            value_ints=common_shape
                        )
                    ),
                )
            )

        if (value := ir.convenience.get_const_tensor(z)) is not None:
            return op.Add(
                op.MatMul(
                    x,
                    op.Expand(
                        y,
                        op.Constant(
                            value_ints=(
                                common_shape := np.broadcast_shapes(
                                    y.shape, z.shape
                                )
                            )
                        )
                    )
                ),
                op.Constant(
                    value=ir.tensor(
                        np.matmul(
                            ir.convenience.get_const_tensor(x),
                            np.broadcast_to(
                                value.numpy(),
                                common_shape
                            )
                        )
                    )
                ),
            )

        raise RuntimeError(
            f"Inconsistent rewrite context: either {x} or {y} must be constant"
        )


@tolerance
class ReorderMatMulReverseDistributiveLhs_v1(RewriteRule, Verify):
    """Reorder MatMul according to reversed left distributivity."""

    @staticmethod
    def pattern(op, x, y, z):
        return op.Add(op.MatMul(x, y), op.MatMul(x, z))

    @staticmethod
    def check(context, x, y, z):
        # Reorder configurations pulling out a common term or bringing constants
        # together, decreasing the number of operations:
        #   xy + xz -> x (y + z)    3 -> 2 operations
        #   cy + cz -> c (y + z)    3 -> 2 operations, constant left
        #   xy + xc -> x (y + c)    3 -> 2 operations
        #   xc + xz -> x (c + z)    3 -> 2 operations
        #   xc + xc -> x (c + c)    3 -> 1 operations, constant foldable right
        # Note: This is the reverse of ReorderMatMulDistributiveLhs_v1
        return not ReorderMatMulDistributiveLhs_v1.check(context, x, y, z)

    @staticmethod
    def rewrite(op, x, y, z):
        return op.MatMul(x, op.Add(y, z))


@tolerance
class ReorderMatMulDistributiveRhs_v1(RewriteRule, Verify):
    """Reorder MatMul according to right distributivity."""

    @staticmethod
    def pattern(op, x, y, z):
        return op.MatMul(op.Add(x, y), z)

    @staticmethod
    def check(context, x, y, z):
        # Dynamic or missing shapes would lead to endless back and forth between
        # forward and reverse distributive rules (if we do not force constant
        # folding as part of every cleanup)
        if x.shape is None or x.shape.is_dynamic():
            if y.shape is None or y.shape.is_dynamic():
                return False

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
    def rewrite(op, x, y, z):
        # Avoid endless back and forth between forward and reverse distributive
        # rules by statically expanding and computing the constant side.
        if (value := ir.convenience.get_const_tensor(x)) is not None:
            return op.Add(
                op.Constant(
                    value=ir.tensor(
                        np.matmul(
                            np.broadcast_to(
                                value.numpy(),
                                common_shape := np.broadcast_shapes(
                                    x.shape, y.shape
                                )
                            ),
                            ir.convenience.get_const_tensor(z)
                        )
                    )
                ),
                op.MatMul(
                    op.Expand(
                        y,
                        op.Constant(
                            value_ints=common_shape
                        )
                    ),
                    z
                )
            )

        if (value := ir.convenience.get_const_tensor(y)) is not None:
            return op.Add(
                op.MatMul(
                    op.Expand(
                        x,
                        op.Constant(
                            value_ints=(
                                common_shape := np.broadcast_shapes(
                                    x.shape, y.shape
                                )
                            )
                        )
                    ),
                    z
                ),
                op.Constant(
                    value=ir.tensor(
                        np.matmul(
                            np.broadcast_to(
                                value.numpy(),
                                common_shape
                            ),
                            ir.convenience.get_const_tensor(z)
                        )
                    )
                ),
            )

        raise RuntimeError(
            f"Inconsistent rewrite context: either {x} or {y} must be constant"
        )


@tolerance
class ReorderMatMulReverseDistributiveRhs_v1(RewriteRule, Verify):
    """Reorder MatMul according to reversed right distributivity."""

    @staticmethod
    def pattern(op, x, y, z):
        return op.Add(op.MatMul(x, z), op.MatMul(y, z))

    @staticmethod
    def check(context, x, y, z):
        # Reorder configurations pulling out a common term or bringing constants
        # together, decreasing the number of operations:
        #   xz + yz -> (x + y) z    3 -> 2 operations
        #   xc + yc -> (x + y) c    3 -> 2 operations, constant right
        #   xz + cz -> (x + c) z    3 -> 2 operations
        #   cz + yz -> (c + y) z    3 -> 2 operations
        #   cz + cz -> (c + c) z    3 -> 1 operations, constant foldable left
        # Note: This is the reverse of ReorderMatMulDistributiveRhs_v1
        return not ReorderMatMulDistributiveRhs_v1.check(context, x, y, z)

    @staticmethod
    def rewrite(op, x, y, z):
        return op.MatMul(op.Add(x, y), z)


class ReorderDistributiveLoop_v1(Sequential):
    """Exhaustively apply distributive reordering transformations."""

    passes = [
        ReorderDistributiveLhs_v1,
        ReorderReverseDistributiveLhs_v1,
        ReorderDistributiveRhs_v1,
        ReorderReverseDistributiveRhs_v1,
        ReorderMatMulDistributiveLhs_v1,
        ReorderMatMulReverseDistributiveLhs_v1,
        ReorderMatMulDistributiveRhs_v1,
        ReorderMatMulReverseDistributiveRhs_v1
    ]

    exhaustive = True
