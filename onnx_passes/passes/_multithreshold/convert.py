from onnx_passes.passes._base import RewriteRule
from onnx_passes.passes._verify import Verify, tolerance

from onnx_passes.ops import DOMAIN as CUSTOM_DOMAIN

import onnx_ir as ir


@tolerance
class RewriteRoundAsMultiThreshold_v1(RewriteRule, Verify):
    """Rewrite rounding with bounds as a MultiThreshold function."""

    @staticmethod
    def pattern(op, x, minimum, maximum):
        return op.Round(op.Min(op.Max(x, minimum), maximum))

    @staticmethod
    def check(context, x, minimum, maximum):
        if ir.convenience.get_const_tensor(minimum) is not None:
            if ir.convenience.get_const_tensor(maximum) is not None:
                return ir.convenience.get_const_tensor(x) is None
        return False

    @staticmethod
    def rewrite_v13(op, x, minimum, maximum):
        return op.Add(
            # Threshold reduction summing the contributions of the individual
            # steps at each threshold.
            op.ReduceSum(
                op.CastLike(
                    op.And(
                        op.GreaterOrEqual(
                            # Append the thresholding dimension to the input
                            # tensor which is later removed by the reduction.
                            op.Unsqueeze(
                                x,
                                op.Constant(value_ints=[-1])
                            ),
                            thresholds := op.Expand(
                                # Thresholds at the decision boundary for Round
                                # at x.5 with round to even correction via Ulp.
                                thresholds := op.Where(
                                    # Apply the round to even correction to even
                                    # positive and odd negative steps, i.e, +2.5
                                    # rounds down to 2.0 and -1.5 down to -2.0.
                                    op.Or(
                                        op.And(
                                            op.GreaterOrEqual(
                                                # Enumerate all x.5 over the
                                                # maximal extend of the range
                                                steps := op.Round(
                                                    op.Range(
                                                        op.ReduceMin(
                                                            op.Round(
                                                                minimum
                                                            )
                                                        ),
                                                        op.ReduceMax(
                                                            op.Round(maximum)
                                                        ),
                                                        op.Constant(
                                                            value_float=1.0
                                                        )
                                                    )
                                                ),
                                                op.Constant(value_float=0.0)
                                            ),
                                            even := op.Equal(
                                                op.Floor(
                                                    op.Mod(
                                                        op.Abs(steps),
                                                        op.Constant(
                                                            value_float=2.0
                                                        ),
                                                        fmod=1
                                                    )
                                                ),
                                                op.Constant(value_float=0.0)
                                            )
                                        ),
                                        op.And(
                                            op.Less(
                                                steps,
                                                op.Constant(value_float=0.0)
                                            ),
                                            # Note: This will be at an odd x.5
                                            # after adding the .5
                                            even
                                        ),
                                    ),
                                    op.Add(
                                        # Shift from integer steps to x.5 steps
                                        # to roughly get the decision boundary
                                        # of the rounding function.
                                        thresholds := op.Add(
                                            steps,
                                            op.Constant(value_float=0.5)
                                        ),
                                        # Round to even correction: Add smallest
                                        # possible increment, i.e., shift the to
                                        # the next larger representable number.
                                        op.Ulp(
                                            thresholds, _domain=CUSTOM_DOMAIN
                                        ),
                                    ),
                                    thresholds
                                ),
                                # Expand to axes covered by minimum and maximum
                                # to not lose the shape contributed by these.
                                op.Shape(
                                    op.Min(
                                        op.Max(
                                            thresholds,
                                            op.Unsqueeze(
                                                minimum,
                                                op.Constant(value_ints=[-1])
                                            )
                                        ),
                                        op.Unsqueeze(
                                            maximum,
                                            op.Constant(value_ints=[-1])
                                        )
                                    )
                                )
                            )
                        ),
                        # Disable contributions from all steps that are out of
                        # the (minimum,maximum) range for each individual axis
                        op.And(
                            op.GreaterOrEqual(
                                thresholds,
                                op.Round(
                                    op.Unsqueeze(
                                        minimum,
                                        op.Constant(value_ints=[-1])
                                    )
                                )
                            ),
                            op.LessOrEqual(
                                thresholds,
                                op.Round(
                                    op.Unsqueeze(
                                        maximum,
                                        op.Constant(value_ints=[-1])
                                    )
                                )
                            ),
                        )
                    ),
                    x
                ),
                # Reduce over the thresholding dimension and delete this axis
                # afterward to restore the original input shape.
                op.Constant(value_ints=[-1]), keepdims=0
            ),
            # Output bias accounting for the minimum shifting where we start
            # counting steps: Shift the output up/down to have f(0)=0.
            op.Round(minimum)
        )
