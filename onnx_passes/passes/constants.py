# ir.Model, ir.passes.PassResult, ir.from_proto, ir.to_proto, ...
import onnx_ir as ir

# Constant folding pass build into ONNX IR and ONNX Script
from onnxscript.optimizer import fold_constants
# Pattern-based graph rewriting implemented in ONNX Script
from onnxscript.rewriter import RewritePass
from onnxscript.rewriter.pattern import RewriteRule

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes


# Performs constant folding on the entire model graph
@passes.verify.equality
@passes.register("fold-constants")
class FoldConstants(passes.base.Transformation):
    # Applies the built-in ONNX IR constant folding pass on a deep copy of the
    # model (as we prefer functional passes not modifying the original).
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Make a deep copy of the model on which the constant folding can
        # operate in-place
        model = ir.from_proto(ir.to_proto(model))
        # Run in-place constant folding on deep copy - yields PassResult
        return fold_constants(model)


# Folds constant shape operators on the entire model graph
@passes.verify.equality
@passes.register("fold-constants")
class FoldConstantShapes(passes.base.Transformation):
    # Folds constant Shape operators on a deep copy of the  model (as we prefer
    # functional passes not modifying the original).
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Match a Shape operation applied to a single tensor x
        def pattern(op, x):
            return op.Shape(x)

        # Pattern match conditions checking for non-symbolic shapes - dynamic
        # shapes (or missing shapes) is not supported
        def condition(_, x: ir.Value):
            return x.shape and all(isinstance(dim, int) for dim in x.shape)

        # Replacement pattern inserting a constant of list of integers
        # representing the shape
        def replacement_pattern(op, x):
            return op.Constant(value_ints=list(x.shape))

        # Create a pattern rewrite rule from the input pattern, condition and
        # replacement pattern
        rule = RewriteRule(pattern, replacement_pattern, condition, verbose=0)
        # Apply the rule as the single rule of a rewrite pass on the model copy
        return RewritePass([rule])(ir.from_proto(ir.to_proto(model)))
