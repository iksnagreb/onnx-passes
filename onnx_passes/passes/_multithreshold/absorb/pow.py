from onnx_passes.passes._base import RewriteRuleSetTemplate
from onnx_passes.passes._verify import Verify

import onnx_ir as ir


class AbsorbPowIntoComparison_v1(RewriteRuleSetTemplate, Verify):
    """Rewrite comparisons to inline and partially solve Pow for x on lhs.

    Derived by inlining solutions to x^a <=> c into the graph, considering all
    possible cases for {x, c, a} {>, <, =} 0 and integer vs. fractional powers.
    """

    patterns = (
        lambda op: op.Greater,
        lambda op: op.Less,
        lambda op: op.Equal,
        lambda op: op.GreaterOrEqual,
        lambda op: op.LessOrEqual,
    )

    @staticmethod
    def pattern(partial, op, x, a, c):
        return partial(op)(op.Pow(x, a), c)

    @staticmethod
    def check(context, x, a, c):
        if ir.convenience.get_const_tensor(a) is not None:
            if ir.convenience.get_const_tensor(c) is not None:
                return ir.convenience.get_const_tensor(x) is None

        return False

    @staticmethod
    def rewrite(partial, op, x, a, c):
        # Input 'x' and constant 'c' are of the same type. Constant 'a' might be
        # of a different type. Depending on whether 'a' is restricted to integer
        # types, the replacement pattern can be simplified.
        #
        # Similarly, the signedness of 'x', 'c' and 'a' (if unsigned), can
        # simplify the replacement pattern.
        a_is_integer_power = op.Constant(
            value=ir.tensor(True)
        )

        if not a.dtype.is_integer():
            a_is_integer_power = op.Equal(
                op.Round(a),
                a
            )

        a_is_positive = op.Constant(
            value=ir.tensor(True)
        )

        if a.dtype.is_signed():
            a_is_positive = op.GreaterOrEqual(
                a,
                op.CastLike(
                    op.Constant(value_float=0.0),
                    a
                )
            )

        c_is_positive = op.Constant(
            value=ir.tensor(True)
        )

        x_is_positive = op.Constant(
            value=ir.tensor(True)
        )

        if c.dtype.is_signed():
            c_is_positive = op.GreaterOrEqual(
                c,
                op.CastLike(
                    op.Constant(value_float=0.0),
                    c
                )
            )

            x_is_positive = op.GreaterOrEqual(
                x,
                op.CastLike(
                    op.Constant(value_float=0.0),
                    x
                )
            )

        # Resolve the (in-)equality into branches depending on type and sign of
        # the constants 'a' and 'c'.
        return op.Where(
            # Handle only instances with positive powers 'a' here as branching
            # conditions for reciprocal (negative powers) is already handled.
            a_is_positive,
            op.Where(
                # Immediately simplify x^0 = 1 and 0^0 = 1 to a constant
                # comparison
                a_is_zero := op.Equal(
                    a,
                    op.CastLike(
                        op.Constant(value_float=0.0),
                        a
                    )
                ),
                partial(op)(
                    op.CastLike(
                        op.Constant(value_float=1.0),
                        x
                    ),
                    c
                ),
                # Root extraction now depends on whether 'a' is an integer power
                # or a fractional power.
                op.Where(
                    a_is_integer_power,
                    # Odd integer powers allow extracting a single signed root
                    # over the entire domain, whereas even powers have roots for
                    # positive 'c' only, but have two solutions +/- c^(1/a).
                    op.Where(
                        a_is_odd_power := op.Equal(
                            op.Mod(
                                op.Abs(a),
                                op.CastLike(
                                    op.Constant(value_float=2.0),
                                    a
                                ),
                                fmod=1
                            ),
                            op.CastLike(
                                op.Constant(value_float=1.0),
                                a
                            )
                        ),
                        partial(op)(
                            x,
                            op.Mul(
                                positive_root := op.Pow(
                                    op.Abs(c),
                                    op.Reciprocal(
                                        op.Cast(
                                            # Avoid dividing by zero for a = 0
                                            op.Where(
                                                a_is_zero,
                                                op.CastLike(
                                                    op.Constant(
                                                        value_float=1.0
                                                    ),
                                                    a
                                                ),
                                                a
                                            ),
                                            # Constant in double precision. The
                                            # result will be of the same type as
                                            # the constant 'c' and input 'x'.
                                            to=ir.DataType.DOUBLE
                                        )
                                    )
                                ),
                                op.Sign(c)
                            )
                        ),
                        op.Xor(
                            op.And(
                                c_is_positive,
                                # Consider both solutions with flipped direction
                                # of comparison for (in-)equalities.
                                op.Or(
                                    partial(op)(
                                        x,
                                        positive_root
                                    ),
                                    partial(op)(
                                        negative_root := op.Neg(
                                            positive_root
                                        ),
                                        x
                                    )
                                )
                            ),
                            # For all x: x^even >= 0
                            op.Not(c_is_positive)
                        )
                    ),
                    # As floats are dyadic rationals, any fractional power or
                    # root has even denominator and odd numerator and thus only
                    # positive inputs 'x' and constant 'c' are allowed.
                    op.Xor(
                        op.And(
                            op.And(
                                x_is_positive,
                                c_is_positive
                            ),
                            partial(op)(
                                x,
                                positive_root
                            )
                        ),
                        # For all x >= 0: x^fraction >= 0
                        op.And(
                            x_is_positive,
                            op.Not(
                                c_is_positive
                            )
                        )
                    )
                )
            ),
            # Rewrite negative powers 'a' into positive powers to be solved by
            # rules above: 1/(x^a) <=> c -> 1 <=> c x^(a) & conditions...
            op.Where(
                a_is_integer_power,
                op.Where(
                    # Odd negative powers have two non-overlapping decreasing
                    # branches: One contributing solutions when 'x' and 'c' are
                    # of the same sign and one for opposing signs. As these are
                    # non-overlapping solutions, we can Xor the expressions.
                    a_is_odd_power,
                    op.Xor(
                        op.And(
                            op.Or(
                                div_ok := op.Not(
                                    op.Equal(
                                        op.CastLike(
                                            op.Constant(value_float=0.0),
                                            c
                                        ),
                                        c
                                    )
                                ),
                                partial(op)(
                                    op.Mul(
                                        op.CastLike(
                                            op.Constant(value_float=1.0),
                                            c
                                        ),
                                        op.Where(
                                            div_ok,
                                            c,
                                            op.CastLike(
                                                op.Constant(value_float=1.0),
                                                c
                                            )
                                        )
                                    ),
                                    op.Pow(
                                        x,
                                        op.Abs(a)
                                    )
                                ),
                            ),
                            op.Xor(
                                op.And(
                                    x_is_positive,
                                    c_is_positive
                                ),
                                op.And(
                                    op.Not(x_is_positive),
                                    op.Not(c_is_positive)
                                )
                            )
                        ),
                        op.And(
                            op.And(
                                div_ok,
                                partial(op)(
                                    op.Pow(
                                        x,
                                        op.Abs(a)
                                    ),
                                    op.Div(
                                        op.CastLike(
                                            op.Constant(value_float=1.0),
                                            c
                                        ),
                                        op.Where(
                                            div_ok,
                                            c,
                                            op.CastLike(
                                                op.Constant(value_float=1.0),
                                                c
                                            )
                                        )
                                    )
                                )
                            ),
                            op.Xor(
                                op.And(
                                    op.Not(x_is_positive),
                                    c_is_positive
                                ),
                                op.And(
                                    x_is_positive,
                                    op.Not(c_is_positive)
                                )
                            )
                        )
                    ),
                    # Even powers have one solution here, which splits into two
                    # solutions for increasing and decreasing branches according
                    # to rules for positive powers above.
                    partial(op)(
                        op.CastLike(
                            op.Constant(value_float=1.0),
                            x
                        ),
                        op.Mul(
                            op.Pow(
                                x,
                                op.Abs(a)
                            ),
                            c
                        )
                    )
                ),
                # Fractional powers have solutions only for positive inputs x,
                # handle cases for positive/negative 'c' according to rules for
                # positive 'a' above.
                op.And(
                    x_is_positive,
                    op.Or(
                        partial(op)(
                            op.CastLike(
                                op.Constant(value_float=1.0),
                                x
                            ),
                            op.Mul(
                                op.Pow(
                                    x,
                                    op.Abs(a)
                                ),
                                c
                            )
                        ),
                        op.Not(
                            c_is_positive
                        )
                    )
                )
            )
        )
