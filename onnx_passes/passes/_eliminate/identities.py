from onnx_passes.passes._base import RewriteRule, RewriteRuleSet, Transformation
from onnx_passes.passes._verify import Verify

import onnx_ir as ir
import numpy as np

from abc import ABC
from typing import Callable, Any


class EliminateIdentity(RewriteRule, ABC):
    """Template: Eliminates the identity element of the operation."""

    operator: Callable
    identity: Any

    def match_identity(self, _, value: ir.Value):
        """Value level checker for the identity element."""
        if (value := ir.convenience.get_const_tensor(value)) is not None:
            return np.all(value.numpy() == self.identity)
        return False

    def pattern(self, op, x):
        return self.operator(op, x, self.match_identity, _outputs=["out"])

    @staticmethod
    def check(op, x, out):
        return out.shape is not None and out.shape.is_static()

    @staticmethod
    def rewrite(op, x, out):
        return op.Expand(x, op.Constant(value_ints=list(out.shape)))


class EliminateIdentityAdd_v1(EliminateIdentity, Verify):
    """Eliminate addition of the identity element."""

    identity = 0

    @staticmethod
    def operator(op, x, y, *args, **kwargs):
        return op.Add(x, y, *args, **kwargs)

    @property
    def commute(self) -> bool:
        return True


class EliminateIdentitySub_v1(EliminateIdentity, Verify):
    """Eliminate subtraction of the identity element."""

    identity = 0

    @staticmethod
    def operator(op, x, y, *args, **kwargs):
        return op.Sub(x, y, *args, **kwargs)


class EliminateIdentityMul_v1(EliminateIdentity, Verify):
    """Eliminate multiplication of the identity element."""

    identity = 1

    @staticmethod
    def operator(op, x, y, *args, **kwargs):
        return op.Mul(x, y, *args, **kwargs)

    @property
    def commute(self) -> bool:
        return True


class EliminateIdentityDiv_v1(EliminateIdentity, Verify):
    """Eliminate division of the identity element."""

    identity = 1

    @staticmethod
    def operator(op, x, y, *args, **kwargs):
        return op.Div(x, y, *args, **kwargs)


class EliminateIdentityOr_v1(EliminateIdentity, Verify):
    """Eliminate boolean Or of the identity element."""

    identity = False

    @staticmethod
    def operator(op, x, y, *args, **kwargs):
        return op.Or(x, y, *args, **kwargs)

    @property
    def commute(self) -> bool:
        return True


class EliminateIdentityBitwiseOr_v1(EliminateIdentity, Verify):
    """Eliminate bitwise Or of the identity element."""

    identity = 0

    @staticmethod
    def operator(op, x, y, *args, **kwargs):
        return op.BitwiseOr(x, y, *args, **kwargs)

    @property
    def commute(self) -> bool:
        return True


class EliminateIdentityAnd_v1(EliminateIdentity, Verify):
    """Eliminate boolean And of the identity element."""

    identity = True

    @staticmethod
    def operator(op, x, y, *args, **kwargs):
        return op.And(x, y, *args, **kwargs)

    @property
    def commute(self) -> bool:
        return True


class EliminateIdentityBitwiseAnd_v1(EliminateIdentity, Verify):
    """Eliminate bitwise And of the identity element."""

    identity = ~0

    @staticmethod
    def operator(op, x, y, *args, **kwargs):
        return op.BitwiseAnd(x, y, *args, **kwargs)

    @property
    def commute(self) -> bool:
        return True


class EliminateIdentityXor_v1(EliminateIdentity, Verify):
    """Eliminate boolean Xor of the identity element."""

    identity = False

    @staticmethod
    def operator(op, x, y, *args, **kwargs):
        return op.Xor(x, y, *args, **kwargs)

    @property
    def commute(self) -> bool:
        return True


class EliminateIdentityBitwiseXor_v1(EliminateIdentity, Verify):
    """Eliminate bitwise Xor of the identity element."""

    identity = 0

    @staticmethod
    def operator(op, x, y, *args, **kwargs):
        return op.BitwiseXor(x, y, *args, **kwargs)

    @property
    def commute(self) -> bool:
        return True


class EliminateIdentityBitShift_v1(EliminateIdentity, Verify):
    """Eliminate bitshift with the identity element."""

    identity = 0

    @staticmethod
    def operator(op, x, y, *args, **kwargs):
        return op.BitShift(x, y, *args, **kwargs)

    @property
    def commute(self) -> bool:
        return True


def identity_matrix(_, matrix: ir.Value):
    """Value level checker for the identity matrix in any dimensions."""

    # Try to unpack the shapes, raises ValueError if there are not enough
    # dimensions to unpack (identity matrix needs at least 2 dimensions)
    try:
        *_, N, M = matrix.shape
    except ValueError:
        return False

    # The potential identity matrix must be square and match the broadcast NxN
    # identity matrix, i.e., it must be identity in all leading dimensions.
    if (N == M) and (matrix := ir.convenience.get_const_tensor(matrix)):
        return np.all(matrix == np.eye(N, N))

    return False


class EliminateIdentityMatMul_v1(RewriteRuleSet, Verify):
    """Eliminate multiplication of the identity matrix."""

    @staticmethod
    def pattern():
        return [
            lambda op, x, eye: op.MatMul(x, identity_matrix, _outputs=["out"]),
            lambda op, x, eye: op.MatMul(identity_matrix, x, _outputs=["out"]),
        ]

    def check(self):
        return [
            lambda op, x, eye, out: \
                out.shape is not None and out.shape.is_static(),
            lambda op, x, eye, out: \
                out.shape is not None and out.shape.is_static(),
        ]

    @staticmethod
    def rewrite():
        return [
            lambda op, x, eye, out: \
                op.Expand(x, op.Constant(value_ints=list(out.shape))),
            lambda op, x, eye, out: \
                op.Expand(x, op.Constant(value_ints=list(out.shape))),
        ]


class EliminateIdentityCast_v1(RewriteRule, Verify):
    """Eliminates identity Cast where the target type is the input type."""

    @staticmethod
    def pattern(op, x, to):
        return op.Cast(x, to=to)

    @staticmethod
    def check(op, x, to):
        return x.dtype == to.as_int()

    @staticmethod
    def rewrite(op, x, to):
        return op.Identity(x)


class EliminateIdentityCastLike_v1(RewriteRule, Verify):
    """Eliminate identity Cast where the target type is the input type."""

    @staticmethod
    def pattern(op, x, target):
        return op.CastLike(x, target)

    @staticmethod
    def check(op, x, target):
        return x.dtype == target.dtype and target.dtype is not None

    @staticmethod
    def rewrite(op, x, target):
        return op.Identity(x)


class EliminateIdentityExpand_v1(RewriteRule, Verify):
    """Eliminate Expand where the target shape is the same as the input."""

    @staticmethod
    def pattern(op, x, shape):
        return op.Expand(x, shape)

    @staticmethod
    def check(op, x, shape):
        if x.shape is not None and x.shape.is_static():
            if (shape := ir.convenience.get_const_tensor(shape)) is not None:
                return tuple(shape.numpy()) == x.shape
        return False

    @staticmethod
    def rewrite(op, x, shape):
        return op.Identity(x)


class EliminateIdentityReshape_v1(RewriteRule, Verify):
    """Eliminate Reshape where the target shape is the same as the input."""

    @staticmethod
    def pattern(op, x, shape):
        return op.Reshape(x, shape)

    @staticmethod
    def check(op, x, shape):
        if x.shape is not None and x.shape.is_static():
            if (shape := ir.convenience.get_const_tensor(shape)) is not None:
                return tuple(shape.numpy()) == x.shape
        return False

    @staticmethod
    def rewrite(op, x, shape):
        return op.Identity(x)


class EliminateIdentityTranspose_v1(RewriteRule, Verify):
    """Eliminate identity transpose operations, i.e., identity permutations."""

    @staticmethod
    def pattern(op, x, perm):
        return op.Transpose(x, perm=perm)

    @staticmethod
    def check(op, x, perm):
        if perm is not None and perm.as_ints() is not None:
            return np.all(perm.as_ints() == tuple(range(len(perm.as_ints()))))
        return False

    @staticmethod
    def rewrite(op, x, perm):
        return op.Identity(x)


class EliminateIdentitySlice_v1(RewriteRule):
    """Eliminate identity Slice operations."""

    @staticmethod
    def pattern(op, x, starts, ends, axes, steps):
        return op.Slice(x, starts, ends, axes, steps, _outputs=["y"])

    @staticmethod
    def check(op, x, starts, ends, axes, steps, y):
        # Constant steps to check for reversal of elements
        if (steps := ir.convenience.get_const_tensor(steps)) is not None:
            # Static and identical input and output shapes
            if x.shape is not None and x.shape.is_static():
                if y.shape is not None and y.shape.is_static():
                    if tuple(x.shape.numpy()) == tuple(y.shape.numpy()):
                        # Slicing backwards might keep the shape but rearrange
                        # the content, this should be rejected
                        return np.all(steps.numpy() == 1)
        return False

    @staticmethod
    def rewrite(op, x, starts, ends, axes, steps, y):
        return op.Identity(x)


# Common identity elimination pass build into ONNX IR and ONNXScript
from onnx_ir.passes.common import IdentityEliminationPass


class EliminateIdentity_v1(Transformation):
    """Eliminate Identity operators where applicable."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return IdentityEliminationPass()(model)
