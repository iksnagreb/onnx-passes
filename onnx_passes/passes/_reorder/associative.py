from onnx_passes.passes._base import RewriteRuleSetTemplate
from onnx_passes.passes._verify import Verify, tolerance

import onnx_ir as ir


@tolerance
class ReorderAssociative_v1(RewriteRuleSetTemplate, Verify):
    """Reorder associative operations to move nesting to the right."""

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
        # Note: Even though matrix multiplication is associative, MatMul
        # broadcasting is not. Thus, it is not included here.
    )

    @staticmethod
    def pattern(partial, op, x, y, z):
        return partial(op)(partial(op)(x, y), z)

    @staticmethod
    def check(context, x, y, z):
        # Do not reorder configurations which already group or isolate constants
        #   (x . x) . c -> isolated constant on the right
        #   (c . c) . x -> constant foldable on the left
        x = ir.convenience.get_const_tensor(x) is not None
        y = ir.convenience.get_const_tensor(y) is not None
        z = ir.convenience.get_const_tensor(z) is not None

        if (x, y, z) in {(False, False, True), (True, True, False)}:
            return False

        return True

    @staticmethod
    def rewrite(partial, op, x, y, z):
        return partial(op)(x, partial(op)(y, z))


@tolerance
class ReorderReverseAssociative_v1(RewriteRuleSetTemplate, Verify):
    """Reorder associative operations to move nesting to the left."""

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
        # Note: Even though matrix multiplication is associative, MatMul
        # broadcasting is not. Thus, it is not included here.
    )

    @staticmethod
    def pattern(partial, op, x, y, z):
        return partial(op)(x, partial(op)(y, z))

    @staticmethod
    def check(context, x, y, z):
        # Reorder configurations which group or isolate constants
        #   x . (x . c) -> (x . x) . c      isolated constant on the right
        #   c . (c . x) -> (c . c) . x      constant foldable on the left
        # Note: This is the reverse of ReorderAssociative_v1
        return not ReorderAssociative_v1.check(context, x, y, z)

    @staticmethod
    def rewrite(partial, op, x, y, z):
        return partial(op)(partial(op)(x, y), z)
