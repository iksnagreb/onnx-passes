# Use ONNX Script for creating test models
from onnxscript import script, opset18 as op, FLOAT

# Base class/template for deriving pass test cases
from onnx_passes.tests.base import PassesTestBase

from onnx_passes.passes.streamline.algebraic.numeric import GroupAdd
from onnx_passes.passes.streamline.algebraic.numeric import EliminateNeg

# For generating test inputs
import numpy as np


class TestGroupConstantAdd(PassesTestBase):
    __passes__ = [GroupAdd]

    @staticmethod
    @script(default_opset=op)
    def __model__(x: FLOAT) -> FLOAT:
        return op.Add(
            op.Add(x, op.Constant(value_float=0.2)),
            op.Constant(value_float=1.4)
        )

    @staticmethod
    @script(default_opset=op)
    def __expected__(x: FLOAT) -> FLOAT:
        return op.Add(
            x,
            op.Add(op.Constant(value_float=0.2), op.Constant(value_float=1.4))
        )

    @staticmethod
    def __inputs__():
        return [np.array(0.5, dtype=np.float32)]


class TestGroupNonConstantAdd(PassesTestBase):
    __passes__ = [GroupAdd]

    @staticmethod
    @script(default_opset=op)
    def __model__(x: FLOAT, y: FLOAT) -> FLOAT:
        return op.Add(op.Add(op.Constant(value_float=0.2), x), y)

    @staticmethod
    @script(default_opset=op)
    def __expected__(x: FLOAT, y: FLOAT) -> FLOAT:
        return op.Add(op.Constant(value_float=0.2), op.Add(x, y))

    @staticmethod
    def __inputs__():
        return [np.array(0.5, dtype=np.float32),
                np.array(1.5, dtype=np.float32)]


class TestEliminateNeg(PassesTestBase):
    __passes__ = [EliminateNeg]

    @staticmethod
    @script(default_opset=op)
    def __model__(x: FLOAT) -> FLOAT:
        return op.Neg(op.Neg(x))

    @staticmethod
    @script(default_opset=op)
    def __expected__(x: FLOAT) -> FLOAT:
        return x

    @staticmethod
    def __inputs__():
        return [np.array(0.5, dtype=np.float32)]
