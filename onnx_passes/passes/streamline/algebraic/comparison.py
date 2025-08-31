# ir.Model, ir.DataType, ir.passes.PassResult, ir.from_proto, ir.to_proto, ...
import onnx_ir as ir

# Algebraic properties as transformation templates
from onnx_passes.passes.streamline.algebraic._properties import (
    _Associative,
    _Commutative,
    _Distributive,
    _Involution,
    _Idempotence,
    _Absorption,
    _Annihilator,
)

# All algebraic passes are transformations derived from pattern-based rewrite
# rules
from onnx_passes.passes.base import Transformation, RewriteRulePass, \
    RewriteRuleSetPass

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes


# ==============================================================================
# Transformations derived from templates by specializing basic algebraic
# properties relating comparison operators ==, !=, >, >=, < and <=
# ==============================================================================

# TODO: Exploit the converse to only have to deal with either (>, >=) or (<, <=)
#  operators?


# ==============================================================================
# Other properties relating comparison operators ==, !=, >, >=, < and <=: ...
# ==============================================================================

# Expands a constant of True to the shape of the input x
def true_like(op, x):
    return op.Expand(
        op.Cast(op.Constant(value_int=1), to=ir.DataType.BOOL), op.Shape(x)
    )


# Expands a constant of False to the shape of the input x
def false_like(op, x):
    return op.Expand(
        op.Cast(op.Constant(value_int=0), to=ir.DataType.BOOL), op.Shape(x)
    )


# Eliminates all instances of comparing a tensor to itself, which yields a
# constant, either True (x == x, x <= x, x >= x) or False (x != x, x < x, x > x)
@passes.verify.equality
@passes.register("algebraic")
class EliminateComparison(Transformation, RewriteRuleSetPass):
    def pattern(self):
        return [
            lambda op, x: op.Equal(x, x),
            lambda op, x: op.LessOrEqual(x, x),
            lambda op, x: op.GreaterOrEqual(x, x),
            lambda op, x: op.Less(x, x),
            lambda op, x: op.Greater(x, x),
        ]

    def rewrite(self):
        return [
            true_like, true_like, true_like, false_like, false_like
        ]
