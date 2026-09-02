from onnx_passes.passes._base import RewriteRule, Transformation, Sequential
from onnx_passes.passes._verify import Verify, tolerance

from onnx_passes.ops import DOMAIN as CUSTOM_DOMAIN


@tolerance
class FuseScaleIntoMultiThreshold_v1(RewriteRule, Verify):
    """Fuse scale following MultiThreshold into the step weights."""

    @staticmethod
    def pattern(op, x, thresholds, weights, scale):
        return op.Mul(
            op.MultiThreshold(x, thresholds, weights, _domain=CUSTOM_DOMAIN),
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
            _domain=CUSTOM_DOMAIN
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
            op.MultiThreshold(x, thresholds1, weights1, _domain=CUSTOM_DOMAIN),
            op.MultiThreshold(x, thresholds2, weights2, _domain=CUSTOM_DOMAIN),
        )

    @staticmethod
    def rewrite_v15(op, x, thresholds1, weights1, thresholds2, weights2):
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

        return op.MultiThreshold(x, thresholds, weights, _domain=CUSTOM_DOMAIN)


from onnx_passes.passes import _fold_constants


class FuseMultiThresholdLoop_v1(Sequential, Transformation):
    """Exhaustively apply multithreshold fusion transformations."""

    passes = [
        FuseScaleIntoMultiThreshold_v1,
        FuseAddedMultiThresholds_v1,
        _fold_constants
    ]

    exhaustive = True
