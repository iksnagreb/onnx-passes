from onnx_passes.passes._base import RewriteRule, Transformation, Sequential
from onnx_passes.passes._verify import Verify, tolerance

from onnx_passes.passes._unbroadcast.elementwise import unbroadcast

from onnx_passes.ops import DOMAIN as CUSTOM_DOMAIN

import onnx_ir as ir
import numpy as np

from onnxscript.rewriter import pattern
from onnxscript.rewriter.pattern import OrValue


def _multithreshold_segments(op, x, y, a1, a2, c1, c2):
    """Pattern matching threshold segments (at least two) connected via Xor."""

    # Match at least two segments of threshold comparisons with optional boolean
    # coefficients.
    return op.Xor(
        lhs := OrValue([
            op.GreaterOrEqual(x, c1),
            # Allow coefficients to commute but not the Xor
            # connective, as threshold segments need to be
            # sorted to rewrite the Xor as a difference.
            OrValue([
                op.And(
                    a1,
                    op.GreaterOrEqual(x, c1)
                ),
                op.And(
                    op.GreaterOrEqual(x, c1),
                    a1,
                ),
            ])
        ]),
        OrValue([
            rhs := OrValue([
                op.GreaterOrEqual(x, c2),
                # Allow coefficients to commute but not the Xor
                # connective, as threshold segments need to be
                # sorted to rewrite the Xor as a difference.
                OrValue([
                    op.And(
                        a2,
                        op.GreaterOrEqual(x, c2)
                    ),
                    op.And(
                        op.GreaterOrEqual(x, c2),
                        a2,
                    ),
                ])
            ]),
            # Recursive application along chains of Xor'ed segments
            # or termination in some other operation or constant.
            op.Xor(rhs, y, _outputs=["z"])
        ])
    )


def _match_multithreshold_segments(model: ir.Model, node: ir.Node):
    """Match pattern of multithreshold segments in mode rooted at node."""
    rule = pattern.Pattern(_multithreshold_segments)
    return rule.match(model, model.graph, node, check_nodes_are_removable=False)


def _check_multithreshold_segments(context, x, y, a1, a2, c1, c2, z=None):
    """Check sorting of threshold segments recursively."""

    # Coefficients a1, a2 are optional, if not present they are implicitly
    # assumed to be True as And(True, x) = x.
    if a1 is not None:  # noqa: Duplicate, see SortConstantComparison_v2
        if (a1 := ir.convenience.get_const_tensor(a1)) is not None:
            a1 = a1.numpy()
        else:
            return False
    else:
        a1 = np.asarray(True)

    if a2 is not None:
        if (a2 := ir.convenience.get_const_tensor(a2)) is not None:
            a2 = a2.numpy()
        else:
            return False
    else:
        a2 = np.asarray(True)

    # Comparison constants are not optional, both must be present to decide
    # whether we need to continue swapping neighboring comparisons.
    if (c1 := ir.convenience.get_const_tensor(c1)) is None:
        return False

    if (c2 := ir.convenience.get_const_tensor(c2)) is None:
        return False

    # If not extending the pattern recursively, stop with checking whether
    # weighted threshold segments are sorted.
    if z is None:
        return np.all((a1 >= a2) & ((a1 != a2) | (c1.numpy() <= c2.numpy())))

    # Local pattern matching to recursively extend the context along the chain
    # of threshold segments without ensuring removability of the matched nodes
    # as they might be part of other, unrelated patterns as well.
    if match := _match_multithreshold_segments(context.model, z.producer()):
        return _check_multithreshold_segments(context, **match.bindings)

    # Threshold segments are either not constant, not sorted or the pattern
    # terminates in some non-thresholding partial pattern.
    return False


@tolerance
class RewriteMultiThresholdXorAsSum_v1(RewriteRule, Verify):
    """Rewrite multithreshold segments in Xor'ed representation as a sum."""

    @staticmethod
    def pattern_v13(op, x, y, a1, a2, c1, c2, dtype, weights, axes):
        return op.ReduceSum(
            OrValue([
                steps := op.Cast(
                    # Match a tree of at least two threshold segments c1,c2 with
                    # optional weights a1,a2, potentially recursing into y.
                    _multithreshold_segments(op, x, y, a1, a2, c1, c2), to=dtype
                ),
                # Optional step weights shared by all threshold segments,
                # explicitly allowed to commute.
                OrValue([
                    op.Mul(weights, steps),
                    op.Mul(steps, weights)
                ])
            ]),
            axes,
            # MultiThreshold: Always delete the threshold axis and never noop,
            # but axes should also not be empty.
            keepdims=0,
            noop_with_empty_axes=0
        )

    @staticmethod
    def check_v13(context, x, y, a1, a2, c1, c2, dtype, weights, axes, z=None):
        return _check_multithreshold_segments(context, x, y, a1, a2, c1, c2, z)

    @staticmethod
    def rewrite_v13(op, x, y, a1, a2, c1, c2, dtype, weights, axes, z=None):
        # Coefficients a1,a2 are optional, if not present they are implicitly
        # assumed to be True as And(True, x) = x. Comparison constants are not
        # optional and always present as guaranteed by the match condition.
        if a1 is None:
            a1 = op.Constant(value=ir.tensor(True))

        if a2 is None:
            a2 = op.Constant(value=ir.tensor(True))

        # Optional step weights shared by all threshold segments are implicitly
        # assumed to be 1 if not present but inserted explicitly here to combine
        # them with the per-segment weights.
        if weights is None:
            weights = op.Cast(op.Constant(value_float=1.0), to=dtype)

        # Rewrite the first two threshold segments joined by Xor into a direct
        # difference: if lhs >= rhs then Xor(lhs,rhs) = Sub(lhs,rhs)
        x = op.Add(
            op.ReduceSum(
                op.Mul(
                    op.Cast(
                        op.GreaterOrEqual(x, c1),
                        to=dtype
                    ),
                    op.Mul(
                        op.CastLike(
                            a1,
                            weights
                        ),
                        weights
                    )
                ),
                axes,
                # MultiThreshold: Always delete the threshold axis and never
                # noop, but axes should also not be empty.
                keepdims=0,
                noop_with_empty_axes=0
            ),
            op.ReduceSum(
                op.Mul(
                    op.Cast(
                        op.GreaterOrEqual(x, c2),
                        to=dtype
                    ),
                    op.Neg(
                        op.Mul(
                            op.CastLike(
                                a2,
                                weights
                            ),
                            weights
                        )
                    )
                ),
                axes,
                # MultiThreshold: Always delete the threshold axis and never
                # noop, but axes should also not be empty.
                keepdims=0,
                noop_with_empty_axes=0
            )
        )

        # If the pattern tail y has been matched, reinsert this part to be
        # matched again
        if y is not None:
            return op.Add(
                x,
                # Reinsert the recursive subpattern y, pushing sum reductions up
                # the tree of threshold segments. Do to the match condition, it
                # is guaranteed that this rewrite rule will match again.
                op.ReduceSum(
                    op.Mul(
                        op.Cast(
                            y,
                            to=dtype
                        ),
                        weights
                    ),
                    axes,
                    # MultiThreshold: Always delete the threshold axis and never
                    # noop, but axes should also not be empty.
                    keepdims=0,
                    noop_with_empty_axes=0
                )
            )

        return x


@tolerance
class InferMultiThreshold_v1(RewriteRule, Verify):
    """Infer MultiThreshold operator from GreaterOrEqual-ReduceSum pattern."""

    @staticmethod
    def pattern_v13(op, x, thresholds, weights, shape, axes, dtype):
        return op.ReduceSum(
            OrValue([
                steps := op.Cast(
                    op.GreaterOrEqual(
                        op.Reshape(
                            x,
                            shape
                        ),
                        thresholds
                    ),
                    to=dtype
                ),
                OrValue([
                    op.Mul(weights, steps),
                    op.Mul(steps, weights)
                ])
            ]),
            axes,
            # MultiThreshold: Always delete the threshold axis and never
            # noop, but axes should also not be empty.
            keepdims=0,
            noop_with_empty_axes=0
        )

    @staticmethod
    def check_v13(context, x, thresholds, weights, shape, axes, dtype):
        # Reduce exactly the single last axis expanded by the reshape operation
        if (shape := ir.convenience.get_const_tensor(shape)) is not None:
            if (axes := ir.convenience.get_const_tensor(axes)) is not None:
                if tuple(axes.numpy()) in {(-1,), (len(shape.numpy()) - 1,)}:
                    if shape.numpy()[-1] == 1:
                        # So far MultiThreshold always yields FLOAT, relax this
                        # in some future version...
                        return dtype.as_int() == ir.DataType.FLOAT

        return False

    @staticmethod
    def rewrite_v13(op, x, thresholds, weights, shape, axes, dtype):
        return op.MultiThreshold(
            # The matched pattern unsqueezes the thresholding axis at the end
            # but might also do other reshaping (always static): Unsqueeze is
            # fused into the custom MultiThreshold operator, thus remove it.
            op.Squeeze(
                op.Reshape(
                    x,
                    shape
                ),
                op.Constant(value_ints=[-1])
            ),
            thresholds,
            weights,
            # This is a custom operator implemented in our custom onnx_passes
            # domain. The function will be linked into the model.
            _domain=CUSTOM_DOMAIN
        )


@tolerance
class SortMultiThreshold_v1(RewriteRule, Verify):
    """Sort MultiThreshold thresholds in increasing order."""

    @staticmethod
    def pattern(op, x, thresholds, weights):
        return op.MultiThreshold(x, thresholds, weights, _domain=CUSTOM_DOMAIN)

    @staticmethod
    def check(context, x, thresholds, weights):
        if (thresholds := ir.convenience.get_const_tensor(thresholds)) is None:
            return False

        if (weights := ir.convenience.get_const_tensor(weights)) is None:
            return False

        # Do not sort singleton thresholds (there might not even be the
        # thresholds axis to sort/index along...)
        if (thresholds := thresholds.numpy()).shape[-1:] in {(), (1,)}:
            return False

        # Do not sort again to not end up in infinite loop of sorting the
        # pattern as sorting does not change the structure of the pattern
        return np.any(np.sort(thresholds, axis=-1) != thresholds)

    @staticmethod
    def rewrite(op, x, thresholds, weights):
        # Extract constant parameter tensors as NumPy arrays: according to the
        # match conditions these are never None and safe to access.
        thresholds = ir.convenience.get_const_tensor(thresholds).numpy()  # noqa
        weights = ir.convenience.get_const_tensor(weights).numpy()  # noqa

        # Broadcast thresholds and weights before sorting to make indices
        # compatible. Unbroadcasting will later remove expanded axes.
        thresholds, weights = np.broadcast_arrays(thresholds, weights)

        order = np.argsort(thresholds, axis=-1)

        thresholds = np.take_along_axis(thresholds, order, axis=-1)
        weights = np.take_along_axis(weights, order, axis=-1)

        thresholds = unbroadcast(thresholds)
        weights = unbroadcast(weights)

        # Insert MultiThreshold operator with sorted parameter constants back
        # into the graph
        thresholds = op.Constant(value=ir.tensor(thresholds))
        weights = op.Constant(value=ir.tensor(weights))

        return op.MultiThreshold(x, thresholds, weights, _domain=CUSTOM_DOMAIN)


from onnx_passes.passes import _fold_constants


class NormalizeMultiThresholdLoop_v1(Sequential, Transformation):
    """Exhaustively apply multithreshold normalization transformations."""

    passes = [
        RewriteMultiThresholdXorAsSum_v1,
        InferMultiThreshold_v1,
        SortMultiThreshold_v1,
        _fold_constants
    ]

    exhaustive = True
