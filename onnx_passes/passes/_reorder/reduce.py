from onnx_passes.passes._base import (
    Transformation, Sequential, RewriteRule, RewriteRuleSetTemplate
)
from onnx_passes.passes._verify import Verify, tolerance

import onnx_ir as ir
import numpy as np


@tolerance
class ReorderCommutativeAssociativeReduce_v1(RewriteRuleSetTemplate, Verify):
    """Reorder commutative and associative operations to follow reduction."""

    patterns = (
        lambda op: (op.Add, op.ReduceSum),
        lambda op: (op.Min, op.ReduceMin),
        lambda op: (op.Max, op.ReduceMax),
        lambda op: (op.Mul, op.ReduceProd)
    )

    @staticmethod
    def pattern_v13(partial, op, x, y, axes):
        return partial(op)[1](partial(op)[0](x, y), axes, _outputs=["out"])

    @staticmethod
    def rewrite_v13(partial, op, x, y, axes, out):
        return partial(op)[0](
            partial(op)[1](
                op.Expand(
                    x, op.Shape(partial(op)[0](x, y))
                ),
                axes,
                **out.producer().attributes
            ),
            partial(op)[1](
                op.Expand(
                    y, op.Shape(partial(op)[0](x, y))
                ),
                axes,
                **out.producer().attributes
            )
        )


@tolerance
class ReorderDistributiveReduce_v1(RewriteRuleSetTemplate, Verify):
    """Reorder distributive operations to follow reduction."""

    patterns = (
        lambda op: (op.Mul, op.ReduceSum),
        lambda op: (op.Add, op.ReduceMin),
        lambda op: (op.Add, op.ReduceMax),
        lambda op: (op.Min, op.ReduceMin),
        lambda op: (op.Max, op.ReduceMax),
        lambda op: (op.Max, op.ReduceMin),
        lambda op: (op.Min, op.ReduceMax),
    )

    @staticmethod
    def pattern_v13(partial, op, x, y, axes):
        return partial(op)[1](partial(op)[0](x, y), axes, _outputs=["out"])

    @staticmethod
    def check_v13(context, x, y, axes, out):
        if (axes := ir.convenience.get_const_tensor(axes)) is not None:
            if x.shape is not None and x.shape.is_static():
                if y.shape is not None and y.shape.is_static():
                    # Input y must be singleton in all reduction axes such that
                    # we can expand and reduce the same constant along these.
                    padded = (*((len(x.shape) - len(y.shape)) * [1]), *y.shape)

                    return np.all(np.asarray(padded)[axes.numpy()] == 1)

        return False

    @staticmethod
    def rewrite_v13(partial, op, x, y, axes, out):
        return partial(op)[0](
            partial(op)[1](
                op.Expand(
                    x, op.Shape(partial(op)[0](x, y))
                ),
                axes,
                **out.producer().attributes
            ),
            # Note: The match condition guarantees the same value is expanded
            # along the reduction axes so we can use ReduceMin to remove them.
            op.ReduceMin(
                op.Expand(
                    y, op.Shape(partial(op)[0](x, y))
                ),
                axes,
                **out.producer().attributes
            )
        )

    @property
    def commute(self) -> bool:
        return True


@tolerance
class MoveMulPastReduceMin_v1(RewriteRule, Verify):
    """Reorder elementwise multiplication to follow ReduceMin reduction."""

    @staticmethod
    def pattern_v13(op, x, y, axes):
        return op.ReduceMin(op.Mul(x, y), axes, _outputs=["out"])

    @staticmethod
    def check_v13(context, x, y, axes, out):
        return ReorderDistributiveReduce_v1.check_v13(context, x, y, axes, out)

    @staticmethod
    def rewrite_v13(op, x, y, axes, out):
        return op.Mul(
            op.Where(
                op.Less(
                    y,
                    op.CastLike(
                        op.Constant(value_float=0.0),
                        y
                    )
                ),
                op.ReduceMax(
                    op.Expand(
                        x, op.Shape(op.Mul(x, y))
                    ),
                    axes,
                    **out.producer().attributes
                ),
                op.ReduceMin(
                    op.Expand(
                        x, op.Shape(op.Mul(x, y))
                    ),
                    axes,
                    **out.producer().attributes
                )
            ),
            op.ReduceMin(
                op.Expand(
                    y, op.Shape(op.Mul(x, y))
                ),
                axes,
                **out.producer().attributes
            )
        )


@tolerance
class MoveMulPastReduceMax_v1(RewriteRule, Verify):
    """Reorder elementwise multiplication to follow ReduceMax reduction."""

    @staticmethod
    def pattern_v13(op, x, y, axes):
        return op.ReduceMax(op.Mul(x, y), axes, _outputs=["out"])

    @staticmethod
    def check_v13(context, x, y, axes, out):
        return ReorderDistributiveReduce_v1.check_v13(context, x, y, axes, out)

    @staticmethod
    def rewrite_v13(op, x, y, axes, out):
        return op.Mul(
            op.Where(
                op.Less(
                    y,
                    op.CastLike(
                        op.Constant(value_float=0.0),
                        y
                    )
                ),
                op.ReduceMin(
                    op.Expand(
                        x, op.Shape(op.Mul(x, y))
                    ),
                    axes,
                    **out.producer().attributes
                ),
                op.ReduceMax(
                    op.Expand(
                        x, op.Shape(op.Mul(x, y))
                    ),
                    axes,
                    **out.producer().attributes
                )
            ),
            op.ReduceMax(
                op.Expand(
                    y, op.Shape(op.Mul(x, y))
                ),
                axes,
                **out.producer().attributes
            )
        )


class ReorderReduceLoop_v1(Sequential, Transformation):
    """Exhaustively apply reduce reordering transformations."""

    passes = [
        ReorderCommutativeAssociativeReduce_v1,
        ReorderDistributiveReduce_v1,
        MoveMulPastReduceMin_v1,
        MoveMulPastReduceMax_v1
    ]

    exhaustive = True
