from onnx_passes.passes.base import RewriteRule, Sequential
from onnx_passes.passes.verify import Verify, tolerance

from onnx_passes.ops import DOMAIN as CUSTOM_DOMAIN

import onnx_ir as ir
import numpy as np


@tolerance
class DecomposeIntoUnitStepsMultiThresholds_v1(RewriteRule, Verify):
    """Decompose non-unit step multithreshold functions.

    Each MultiThreshold function is rewritten as a sum over positive or negative
    unit steps (zero steps are also allowed).

    To achieve this, a common factor is extracted and thresholds are replicated
    according to multiplicity.
    """

    @staticmethod
    def pattern(op, x, thresholds, weights):
        return op.MultiThreshold(x, thresholds, weights, _domain=CUSTOM_DOMAIN)

    @staticmethod
    def check(context, x, thresholds, weights):
        if (weights := ir.convenience.get_const_tensor(weights)) is not None:
            return np.any(
                (np.abs(weights.numpy()) != 1) & (np.abs(weights.numpy()) != 0)
            )

        return False

    @staticmethod
    def rewrite(op, x, thresholds, weights):
        # Extract constant parameter tensors as NumPy arrays: according to the
        # match conditions these are never None and safe to access.
        thresholds = ir.convenience.get_const_tensor(thresholds).numpy()  # noqa
        weights = ir.convenience.get_const_tensor(weights).numpy()  # noqa

        # Broadcast thresholds and weights before eliminating to make indices
        # compatible. Unbroadcasting will later remove expanded axes.
        thresholds, weights = np.broadcast_arrays(thresholds, weights)

        shape = thresholds.shape[:-1]

        thresholds_dtype = thresholds.dtype
        weights_dtype = weights.dtype

        # Collect the unique, i.e., non-repeated thresholds from flattened
        # channel dimensions
        thresholds = np.reshape(thresholds, (-1, thresholds.shape[-1]))
        weights = np.reshape(weights, (-1, weights.shape[-1]))

        unique, scales = [], []

        for ts, ws in zip(thresholds, weights):
            channel = {}

            for threshold, weight in zip(ts, ws):
                if threshold not in channel:
                    channel[threshold] = 0.0
                channel[threshold] += weight

            unique.append(channel)

            import math

            scales.append(
                math.gcd(*(round(x / 1e-15) for x in channel.values())) * 1e-15
            )

        for channel, scale in zip(unique, scales):
            for threshold, weight in channel.items():
                channel[threshold] = round(weight / scale)

        # Fill up all channels to the same number of thresholds, inserting some
        # dead thresholds to have homogeneous shapes
        max_steps = int(max(sum(channel.values()) for channel in unique))

        for channel in unique:
            for _ in range(max_steps - int(sum(channel.values()))):
                # Repeat dead copies of the last threshold or zeros if there is
                # no last threshold
                channel.append((channel[-1][0] if channel else 0, 0))

        # Replicate each threshold according to ..., and explicitly keep dead
        # thresholds to have homogeneous shapes
        replicates = []

        for channel in unique:
            for threshold, weight in channel.items():
                if weight:
                    for _ in range(int(abs(weight))):
                        replicates.append((threshold, np.sign(weight)))
                else:
                    replicates.append((threshold, 0))

        # Reconstruct the actual channel shapes of thresholds and weights and
        # recover the original datatypes
        thresholds, weights = np.split(np.asarray(replicates), 2, axis=-1)

        thresholds = thresholds.astype(thresholds_dtype)
        weights = weights.astype(weights_dtype)

        thresholds = np.reshape(thresholds, (*shape, -1))
        weights = np.reshape(weights, (*shape, -1))

        # Rearrange the common shapes to match the output shape and cast to the
        # same type as the weights (also the output type)
        scales = np.asarray(scales).astype(weights_dtype).reshape(shape)

        # Insert MultiThreshold operator with ... parameter constants back
        # into the graph and scale the output by common scales
        thresholds = op.Constant(value=ir.tensor(thresholds))
        weights = op.Constant(value=ir.tensor(weights))
        scales = op.Constant(value=ir.tensor(scales))

        return op.Mul(
            op.MultiThreshold(
                x, thresholds, weights, _domain=CUSTOM_DOMAIN
            ),
            scales
        )


@tolerance
class DecomposeIntoMonotonicMultiThresholds_v1(RewriteRule, Verify):
    """Decompose non-monotonic multithreshold functions.

    Each MultiThreshold function is rewritten as the difference of an increasing
    and a decreasing function.
    """

    @staticmethod
    def pattern(op, x, thresholds, weights):
        return op.MultiThreshold(x, thresholds, weights, _domain=CUSTOM_DOMAIN)

    @staticmethod
    def check(context, x, thresholds, weights):
        if (weights := ir.convenience.get_const_tensor(weights)) is not None:
            return np.any(weights.numpy() < 0)

        return False

    @staticmethod
    def rewrite(op, x, thresholds, weights):
        return op.Sub(
            # Function of the increasing subset of steps: Keep all weights >= 0
            # and set negative steps to zero.
            op.MultiThreshold(
                x,
                thresholds,
                op.Where(
                    op.Greater(
                        weights,
                        op.CastLike(
                            op.Constant(value_float=0.0),
                            weights
                        )
                    ),
                    weights,
                    op.CastLike(
                        op.Constant(value_float=0.0),
                        weights
                    )
                ),
                _domain=CUSTOM_DOMAIN
            ),
            # Function of the decreasing subset of steps: Flip all weights < 0
            # and set positive steps to zero.
            op.MultiThreshold(
                x,
                thresholds,
                op.Where(
                    op.Greater(
                        weights,
                        op.CastLike(
                            op.Constant(value_float=0.0),
                            weights
                        )
                    ),
                    op.CastLike(
                        op.Constant(value_float=0.0),
                        weights
                    ),
                    op.Abs(
                        weights
                    ),
                ),
                _domain=CUSTOM_DOMAIN
            ),
        )


from onnx_passes.passes import fold_constants
from onnx_passes.passes import multithreshold


class DecomposeMultiThresholds_v1(Sequential):
    """Apply multithreshold decomposition transformations."""

    passes = [
        DecomposeIntoUnitStepsMultiThresholds_v1,
        DecomposeIntoMonotonicMultiThresholds_v1,
        # Note: Only selected normalization passes, especially not the threshold
        # deduplication, as it might yield non-unit steps.
        multithreshold.normalize.SortMultiThreshold_v1,
        multithreshold.normalize.EliminateDeadThresholds_v1,
        multithreshold.normalize.UnbroadcastThresholds_v1,
        fold_constants,
    ]

    exhaustive = True
