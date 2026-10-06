from onnx_passes.passes._base import RewriteRuleSet
from onnx_passes.passes._verify import Verify

import onnx_ir as ir
import numpy as np


class AbsorbExpIntoComparison_v1(RewriteRuleSet, Verify):
    """Rewrite comparisons to absorb exponential into constant rhs."""

    @staticmethod
    def pattern():
        return [
            lambda op, x, c: op.Greater(op.Exp(x), c),
            lambda op, x, c: op.Less(op.Exp(x), c),
            lambda op, x, c: op.Equal(op.Exp(x), c),
            lambda op, x, c: op.GreaterOrEqual(op.Exp(x), c),
            lambda op, x, c: op.LessOrEqual(op.Exp(x), c),
        ]

    @staticmethod
    def check():
        return [
            lambda op, x, c: ir.convenience.get_const_tensor(c) is not None,
            lambda op, x, c: ir.convenience.get_const_tensor(c) is not None,
            lambda op, x, c: ir.convenience.get_const_tensor(c) is not None,
            lambda op, x, c: ir.convenience.get_const_tensor(c) is not None,
            lambda op, x, c: ir.convenience.get_const_tensor(c) is not None,
        ]

    @staticmethod
    def rewrite():
        return [
            # Exp(x) > c -> c > 0 & x > Log(c) | c <= 0
            lambda op, x, c: op.Or(
                op.And(
                    log_ok := op.Greater(
                        c,
                        op.CastLike(op.Constant(value_float=0.0), c)
                    ),
                    op.Greater(
                        x,
                        op.Log(
                            op.Where(
                                log_ok,
                                c,
                                op.CastLike(
                                    op.Constant(value_float=1.0),
                                    c
                                )
                            )
                        )
                    )
                ),
                op.LessOrEqual(
                    c,
                    op.CastLike(op.Constant(value_float=0.0), c)
                )
            ),
            # Exp(x) < c -> c > 0 & x < Log(c)
            lambda op, x, c: op.And(
                log_ok := op.Greater(
                    c,
                    op.CastLike(op.Constant(value_float=0.0), c)
                ),
                op.Less(
                    x,
                    op.Log(
                        op.Where(
                            log_ok,
                            c,
                            op.CastLike(
                                op.Constant(value_float=1.0),
                                c
                            )
                        )
                    )
                )
            ),
            # Exp(x) == c -> c > 0 & x == Log(c) | c == 0 & x <= -inf
            lambda op, x, c: op.Or(
                op.And(
                    log_ok := op.Greater(
                        c,
                        op.CastLike(op.Constant(value_float=0.0), c)
                    ),
                    op.Equal(
                        x,
                        op.Log(
                            op.Where(
                                log_ok,
                                c,
                                op.CastLike(
                                    op.Constant(value_float=1.0),
                                    c
                                )
                            )
                        )
                    )
                ),
                op.And(
                    op.Equal(
                        c,
                        op.CastLike(
                            op.Constant(value_float=0.0),
                            c
                        )
                    ),
                    op.Equal(
                        x,
                        op.CastLike(
                            op.Constant(value_float=-np.inf),
                            c
                        )
                    )
                )
            ),
            # Exp(x) >= c -> c > 0 & x >= Log(c) | c <= 0
            lambda op, x, c: op.Or(
                op.And(
                    log_ok := op.Greater(
                        c,
                        op.CastLike(op.Constant(value_float=0.0), c)
                    ),
                    op.GreaterOrEqual(
                        x,
                        op.Log(
                            op.Where(
                                log_ok,
                                c,
                                op.CastLike(
                                    op.Constant(value_float=1.0),
                                    c
                                )
                            )
                        )
                    )
                ),
                op.LessOrEqual(
                    c,
                    op.CastLike(op.Constant(value_float=0.0), c)
                )
            ),

            # Exp(x) <= c -> c > 0 & x <= Log(c) | c == 0 & x == -inf
            lambda op, x, c: op.Or(
                op.And(
                    log_ok := op.Greater(
                        c,
                        op.CastLike(op.Constant(value_float=0.0), c)
                    ),
                    op.LessOrEqual(
                        x,
                        op.Log(
                            op.Where(
                                log_ok,
                                c,
                                op.CastLike(
                                    op.Constant(value_float=1.0),
                                    c
                                )
                            )
                        )
                    )
                ),
                op.And(
                    op.Equal(
                        c,
                        op.CastLike(
                            op.Constant(value_float=0.0),
                            c
                        )
                    ),
                    op.Equal(
                        x,
                        op.CastLike(
                            op.Constant(value_float=-np.inf),
                            c
                        )
                    )
                )
            )
        ]
