from onnx_passes.passes._base import RewriteRule, Transformation, Sequential
from onnx_passes.passes._verify import Verify

import onnx_ir as ir
import numpy as np


class EliminateWhere_v1(RewriteRule, Verify):
    """Eliminates Where with constant all True/False condition input."""

    @staticmethod
    def pattern(op, condition, lhs, rhs):
        return op.Where(condition, lhs, rhs, _outputs=["out"])

    @staticmethod
    def check(op, condition, lhs, rhs, out):
        if condition := ir.convenience.get_const_tensor(condition):
            return np.all(condition.numpy()) or not np.any(condition.numpy())
        return False

    @staticmethod
    def rewrite(op, condition, lhs, rhs, out):
        if np.all(ir.convenience.get_const_tensor(condition).numpy()):
            return op.Expand(lhs, op.Constant(value_ints=list(out.shape)))
        return op.Expand(rhs, op.Constant(value_ints=list(out.shape)))


class EliminateIdentityWhere_v1(RewriteRule, Verify):
    """Eliminate Where with two identical branches."""

    @staticmethod
    def pattern(op, condition, x):
        return op.Where(condition, x, x, _outputs=["out"])

    @staticmethod
    def check(op, condition, lhs, rhs, out):
        if condition := ir.convenience.get_const_tensor(condition):
            return np.all(condition.numpy()) or not np.any(condition.numpy())
        return False

    @staticmethod
    def rewrite(op, condition, x, out):
        return op.Expand(x, op.Constant(value_ints=list(out.shape)))


class EliminateBranchesLoop_v1(Sequential, Transformation):
    """Exhaustively apply branch elimination transformations."""

    passes = [
        EliminateWhere_v1,
        EliminateIdentityWhere_v1,
    ]

    exhaustive = True
