# ir.Model, ir.passes.PassResult, ir.from_proto, ir.to_proto, ...
import onnx_ir as ir

# Unused node removal passes build into ONNX IR
from onnx_ir.passes.common import RemoveUnusedNodesPass

# Constant folding pass build into ONNX IR and ONNX Script
from onnxscript.optimizer import fold_constants

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# Derive Transformations (allowed to modify the graph) from pattern-based
# rewrite rules
from onnx_passes.passes.base import Transformation, RewriteRulePass

# NumPy used during match condition checks to operate on shapes and tensors
import numpy as np


# Performs constant folding on the entire model graph
@passes.verify.equality
@passes.register("fold-constants")
class FoldConstants(Transformation):
    # Applies the built-in ONNX IR constant folding pass on a deep copy of the
    # model (as we prefer functional passes not modifying the original).
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Make a deep copy of the model on which the constant folding can
        # operate in-place
        model = ir.from_proto(ir.to_proto(model))
        # Run in-place constant folding on deep copy - yields PassResult
        modified = fold_constants(model).modified
        # Constant folding might leave unused initializer nodes in the graph
        # which can be removed in-place
        result = RemoveUnusedNodesPass()(model)
        # Combine pass result from both passes to not miss modifications due to
        # unused nodes unrelated to constant folding
        return ir.passes.PassResult(result.model, modified or result.modified)


# Folds constant shape operators on the entire model graph
@passes.verify.equality
@passes.register("fold-constants")
class FoldConstantShapes(Transformation, RewriteRulePass):
    # Match a Shape operation applied to a single tensor x
    def pattern(self, op, x):
        return op.Shape(x)

    # Pattern match conditions checking for non-symbolic shapes - dynamic
    # shapes (or missing shapes) is not supported
    def check(self, _, x: ir.Value):
        return x.shape and all(isinstance(dim, int) for dim in x.shape)

    # Replacement pattern inserting a constant of list of integers
    # representing the shape
    def rewrite(self, op, x):
        return op.Constant(value_ints=list(x.shape))


# Eliminates Where operators if the condition is a constant and always chooses
# the same branch: This rule selects the left hand side if possible
@passes.verify.equality
@passes.register("simplify")
@passes.register("fold-constants")
@passes.register("eliminate-where")
class EliminateWhereLhs(Transformation, RewriteRulePass):
    def pattern(self, op, condition, lhs, rhs):
        return op.Where(condition, lhs, rhs)

    def check(self, op, condition, lhs, rhs):
        if condition := ir.convenience.get_const_tensor(condition):
            return np.all(condition.numpy())
        return False

    def rewrite(self, op, condition, lhs, rhs):
        return lhs


# Eliminates Where operators if the condition is a constant and always chooses
# the same branch: This rule selects the right hand side if possible
@passes.verify.equality
@passes.register("simplify")
@passes.register("fold-constants")
@passes.register("eliminate-where")
class EliminateWhereRhs(Transformation, RewriteRulePass):
    def pattern(self, op, condition, lhs, rhs):
        return op.Where(condition, lhs, rhs)

    def check(self, op, condition, lhs, rhs):
        if condition := ir.convenience.get_const_tensor(condition):
            return np.all(condition.numpy() == False)
        return False

    def rewrite(self, op, condition, lhs, rhs):
        return rhs
