from onnx_passes.passes._base import RewriteRule, Transformation, Sequential
from onnx_passes.passes._verify import Verify

import onnx_ir as ir
import numpy as np

from abc import ABC
from typing import Callable, Any


class EliminateAnnihilator(RewriteRule, ABC):
    """Template: Eliminate the annihilator of the operation."""

    operator: Callable
    annihilator: Any

    def match_annihilator(self, _, value: ir.Value):
        """Value level checker for the annihilator element."""
        if (value := ir.convenience.get_const_tensor(value)) is not None:
            return np.all(value.numpy() == self.annihilator)
        return False

    def pattern(self, op, x):
        return self.operator(op, x, self.match_annihilator, _outputs=["out"])

    @staticmethod
    def check(context, x, out):
        return out.shape is not None and out.shape.is_static()

    def rewrite(self, op, x, out):
        return op.Expand(
            op.CastLike(
                op.Constant(value=ir.tensor(self.annihilator)),
                x
            ),
            op.Constant(value_ints=list(out.shape))
        )


class EliminateAnnihilatorMul_v1(EliminateAnnihilator, Verify):
    """Eliminate multiplication annihilator, i.e., zero"""

    annihilator = 0

    @staticmethod
    def operator(op, x, y, *args, **kwargs):
        return op.Mul(x, y, *args, **kwargs)

    @property
    def commute(self) -> bool:
        return True


class EliminateAnnihilatorAnd_v1(EliminateAnnihilator, Verify):
    """Eliminate boolean And annihilator, i.e., False"""

    annihilator = False

    @staticmethod
    def operator(op, x, y, *args, **kwargs):
        return op.And(x, y, *args, **kwargs)

    @property
    def commute(self) -> bool:
        return True


class EliminateAnnihilatorOr_v1(EliminateAnnihilator, Verify):
    """Eliminate boolean Or annihilator, i.e., True"""

    annihilator = True

    @staticmethod
    def operator(op, x, y, *args, **kwargs):
        return op.Or(x, y, *args, **kwargs)

    @property
    def commute(self) -> bool:
        return True


class EliminateAnnihilatorBitwiseAnd_v1(EliminateAnnihilator, Verify):
    """Eliminate bitwise And annihilator, i.e., all bits zero"""

    annihilator = 0

    @staticmethod
    def operator(op, x, y, *args, **kwargs):
        return op.BitwiseAnd(x, y, *args, **kwargs)

    @property
    def commute(self) -> bool:
        return True


class EliminateAnnihilatorBitwiseOr_v1(EliminateAnnihilator, Verify):
    """Eliminate bitwise Or annihilator, i.e., all bits one"""

    annihilator = ~0  # = 111...1

    @staticmethod
    def operator(op, x, y, *args, **kwargs):
        return op.BitwiseOr(x, y, *args, **kwargs)

    @property
    def commute(self) -> bool:
        return True


class EliminateAnnihilatorMin_v1(EliminateAnnihilator, Verify):
    """Eliminate minimum annihilator, i.e., -infinity"""

    annihilator = -np.inf

    @staticmethod
    def operator(op, x, y, *args, **kwargs):
        return op.Min(x, y, *args, **kwargs)

    @property
    def commute(self) -> bool:
        return True


class EliminateAnnihilatorMax_v1(EliminateAnnihilator, Verify):
    """Eliminate maximum annihilator, i.e., +infinity"""

    annihilator = +np.inf

    @staticmethod
    def operator(op, x, y, *args, **kwargs):
        return op.Max(x, y, *args, **kwargs)

    @property
    def commute(self) -> bool:
        return True


class EliminateAnnihilatorLoop_v1(Sequential, Transformation):
    """Exhaustively apply annihilator elimination transformations."""

    passes = [
        EliminateAnnihilatorMul_v1,
        EliminateAnnihilatorAnd_v1,
        EliminateAnnihilatorOr_v1,
        EliminateAnnihilatorBitwiseAnd_v1,
        EliminateAnnihilatorBitwiseOr_v1,
        EliminateAnnihilatorMin_v1,
        EliminateAnnihilatorMax_v1
    ]

    exhaustive = True
