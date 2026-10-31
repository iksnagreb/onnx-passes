from onnx_passes.passes.base import RewriteRule, Sequential
from onnx_passes.passes.verify import Verify, tolerance

from onnx_passes.ops import DOMAIN as PASSES_DOMAIN

import onnx_ir as ir
import numpy as np


@tolerance
class FuseScaleIntoMultiThreshold_v1(RewriteRule, Verify):
    """Fuse scale following MultiThreshold into the step weights."""

    @staticmethod
    def pattern(op, x, thresholds, weights, scale):
        return op.Mul(
            op.MultiThreshold(x, thresholds, weights, _domain=PASSES_DOMAIN),
            scale
        )

    @staticmethod
    def rewrite_v13(op, x, thresholds, weights, scale):
        return op.MultiThreshold(
            x,
            thresholds,
            op.Mul(
                weights,
                # Scale expands along the thresholding axis to match the weights
                op.Unsqueeze(
                    scale, op.Constant(value_ints=[-1])
                )
            ),
            _domain=PASSES_DOMAIN
        )

    @property
    def commute(self) -> bool:
        return True


@tolerance
class FuseAddedMultiThresholds_v1(RewriteRule, Verify):
    """Fuse two MultiThreshold operators joined via addition."""

    @staticmethod
    def pattern(op, x, thresholds1, weights1, thresholds2, weights2):
        return op.Add(
            op.MultiThreshold(x, thresholds1, weights1, _domain=PASSES_DOMAIN),
            op.MultiThreshold(x, thresholds2, weights2, _domain=PASSES_DOMAIN),
        )

    @staticmethod
    def rewrite_v15(op, x, thresholds1, weights1, thresholds2, weights2):
        # Extend thresholds to common rank by unsqueezing dimensions of size 1
        # from the left of both parameter tensors. Weights are extended when
        # thresholds and weights are broadcast against each other.
        thresholds1 = op.Unsqueeze(
            thresholds1,
            op.Range(
                op.Size(
                    op.Shape(thresholds1)
                ),
                common_rank := op.Max(
                    op.Size(
                        op.Shape(thresholds1)
                    ),
                    op.Size(
                        op.Shape(thresholds2)
                    ),
                    op.Constant(value_int=1)
                ),
                op.Constant(value_int=1),
            )
        )

        thresholds2 = op.Unsqueeze(
            thresholds2,
            op.Range(
                op.Size(
                    op.Shape(thresholds2)
                ),
                common_rank,
                op.Constant(value_int=1),
            )
        )

        # Expand thresholds and weights on either side to common shape such that
        # there is a 1:1 correspondence, required to stack matching amounts when
        # fusing the two operators.
        thresholds1 = op.Expand(
            thresholds1,
            common_shape1 := op.Shape(
                op.Max(
                    thresholds1,
                    op.CastLike(weights1, thresholds1)
                )
            )
        )

        weights1 = op.Expand(weights1, common_shape1)

        thresholds2 = op.Expand(
            thresholds2,
            common_shape2 := op.Shape(
                op.Max(
                    thresholds2,
                    op.CastLike(weights2, thresholds2)
                )
            )
        )

        weights2 = op.Expand(weights2, common_shape2)

        # Fuse the two operators by concatenating the thresholds and weights
        # along the thresholding axis after expanding all leading dimensions.
        thresholds = op.Concat(
            op.Expand(
                thresholds1,
                op.Concat(
                    # Common leading dimensions of thresholds and weights: after
                    # already expanding each side this is the same for all
                    common_shape := op.Max(
                        op.Shape(thresholds1, end=-1),
                        op.Shape(thresholds2, end=-1),
                    ),
                    op.Shape(thresholds1, start=-1),
                    axis=-1
                )
            ),
            op.Expand(
                thresholds2,
                op.Concat(
                    common_shape,
                    op.Shape(thresholds2, start=-1),
                    axis=-1
                )
            ),
            axis=-1
        )

        weights = op.Concat(
            op.Expand(
                weights1,
                op.Concat(
                    common_shape,
                    op.Shape(weights1, start=-1),
                    axis=-1
                )
            ),
            op.Expand(
                weights2,
                op.Concat(
                    common_shape,
                    op.Shape(weights2, start=-1),
                    axis=-1
                )
            ),
            axis=-1
        )

        return op.MultiThreshold(x, thresholds, weights, _domain=PASSES_DOMAIN)


@tolerance
class FuseConsecutiveMultiThresholds_v1(RewriteRule, Verify):
    """Fuses two consecutive MultiThreshold operations into a single operation.
    """

    @staticmethod
    def pattern_v18(op, x, thresholds1, weights1, thresholds2, weights2):
        return op.MultiThreshold(
            op.MultiThreshold(
                x, thresholds1, weights1, _domain=PASSES_DOMAIN
            ),
            thresholds2, weights2, _domain=PASSES_DOMAIN
        )

    @staticmethod
    def check_v18(context, x, thresholds1, weights1, thresholds2, weights2):
        # All thresholds must be sorted to make use of the cumulative sum of the
        # step weights, which only gives a correct sketch of the function if the
        # corresponding thresholds are in increasing order.
        if (t1 := ir.convenience.get_const_tensor(thresholds1)) is None:
            return False

        if np.any(np.sort(t1, axis=-1) != t1):
            return False

        if (t2 := ir.convenience.get_const_tensor(thresholds2)) is None:
            return False

        if np.any(np.sort(t2, axis=-1) != t2):
            return False

        # With non-constant step weights, this transformation gets messy rather
        # quickly - for now restrict this to the constant case...
        if ir.convenience.get_const_tensor(weights1) is not None:
            return ir.convenience.get_const_tensor(weights2) is not None

        return False

    @staticmethod
    def rewrite_v18(op, x, thresholds1, weights1, thresholds2, weights2):
        # The cumulative sum of the padded weights gives the output levels of
        # the first MultiThreshold function.
        levels = op.CumSum(
            op.Pad(
                # Expand the weights to match the thresholds at least along the
                # thresholding, i.e., final, axis.
                op.Expand(
                    weights1,
                    op.Shape(
                        thresholds1, start=-1
                    )
                ),
                # Add zero padding on the left such that the output levels start
                # counting at the zero level
                op.Constant(value_ints=[1, 0]),
                op.CastLike(
                    op.Constant(value_int=0),
                    weights1
                ),
                op.Constant(value_ints=[-1]),
            ),
            op.Constant(value_int=-1)
        )

        # Expand the second MultiThreshold function to accommodate the extra
        # axis introduced by the weights of the first.
        thresholds2 = op.Unsqueeze(
            op.Expand(
                thresholds2,
                op.Constant(value_ints=[1])
            ),
            op.Constant(value_ints=[-2])
        )

        weights2 = op.Unsqueeze(
            op.Expand(
                weights2,
                op.Constant(value_ints=[1])
            ),
            op.Constant(value_ints=[-2])
        )

        # Transform this sketch of the first function according to the second
        # MultiThreshold function.
        levels = op.MultiThreshold(
            levels, thresholds2, weights2, _domain=PASSES_DOMAIN
        )

        # Compute transformed step weights matching the thresholds of the first
        # MultiThreshold function by taking the "derivative" of the transformed
        # output levels, i.e., use a sliding [-1,+1] kernel.
        fused_weights = op.Sub(
            # Shift the levels one to the left, dropping the leftmost padded
            # level - this gives the +1 weight for the derivative kernel.
            op.Slice(
                levels,
                op.Constant(value_ints=[1]),
                op.Shape(
                    levels, start=-1
                ),
                op.Constant(value_ints=[-1]),
            ),
            # Slice of the final level, keeping the leftmost padded level - this
            # gives the -1 weight for the derivative kernel.
            op.Slice(
                levels,
                op.Constant(value_ints=[0]),
                op.Constant(value_ints=[-1]),
                op.Constant(value_ints=[-1]),
            )
        )

        # Bias the output by where the zero level of the first function gets
        # mapped by the second function, i.e., the leftmost transformed level.
        bias = op.Squeeze(
            op.Slice(
                levels,
                op.Constant(value_ints=[+0]),
                op.Constant(value_ints=[+1]),
                op.Constant(value_ints=[-1]),
            ),
            op.Constant(value_ints=[-1]),
        )

        return op.Add(
            op.MultiThreshold(
                x, thresholds1, fused_weights, _domain=PASSES_DOMAIN
            ),
            bias
        )


@tolerance
class FuseAddIntoMultiThreshold_v1(RewriteRule, Verify):
    """Fuse constant addition preceding MultiThreshold into the thresholds.

    This seems redundant, already covered by absorption passes, but additions
    might reappear when fusing chains of MultiThreshold operations.
    """

    @staticmethod
    def pattern(op, x, y, thresholds, weights):
        return op.MultiThreshold(
            op.Add(x, y), thresholds, weights, _domain=PASSES_DOMAIN
        )

    @staticmethod
    def check(context, x, y, thresholds, weights):
        return ir.convenience.get_const_tensor(y) is not None

    @staticmethod
    def rewrite(op, x, y, thresholds, weights):
        return op.MultiThreshold(
            x,
            op.Sub(
                thresholds,
                op.Unsqueeze(
                    y,
                    op.Constant(value_ints=[-1])
                )
            ),
            weights,
            _domain=PASSES_DOMAIN
        )


from onnx_passes.passes import fold_constants
from onnx_passes.passes import multithreshold


class FuseMultiThresholdLoop_v1(Sequential):
    """Exhaustively apply multithreshold fusion transformations."""

    passes = [
        FuseScaleIntoMultiThreshold_v1,
        FuseAddedMultiThresholds_v1,
        FuseConsecutiveMultiThresholds_v1,
        FuseAddIntoMultiThreshold_v1,
        fold_constants,
        multithreshold.normalize,
    ]

    exhaustive = True
