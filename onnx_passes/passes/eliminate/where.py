# ir.Model, ir.passes.PassResult, ir.from_proto, ir.to_proto, ...
import onnx_ir as ir

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# Derive Transformations (allowed to modify the graph) from pattern-based
# rewrite rules
from onnx_passes.passes.base import Transformation, RewriteRulePass

# NumPy used during match condition checks to operate on shapes and tensors
import numpy as np


# Eliminates Where operators if the condition is a constant and always chooses
# the same branch: This rule selects the left hand side if possible
@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-where")
class EliminateWhereLhs(Transformation, RewriteRulePass):
    def pattern(self, op, condition, lhs, rhs):
        return op.Where(condition, lhs, rhs, _outputs=["_out"])

    def check(self, op, condition, lhs, rhs, _out):
        if condition := ir.convenience.get_const_tensor(condition):
            return _out.shape is not None and np.all(condition.numpy())
        return False

    def rewrite(self, op, condition, lhs, rhs, _out):
        return op.Expand(lhs, op.Constant(value_ints=list(_out.shape)))


# Eliminates Where operators if the condition is a constant and always chooses
# the same branch: This rule selects the right hand side if possible
@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-where")
class EliminateWhereRhs(Transformation, RewriteRulePass):
    def pattern(self, op, condition, lhs, rhs):
        return op.Where(condition, lhs, rhs, _outputs=["_out"])

    def check(self, op, condition, lhs, rhs, _out):
        if condition := ir.convenience.get_const_tensor(condition):
            return _out.shape is not None and np.all(condition.numpy() == False)
        return False

    def rewrite(self, op, condition, lhs, rhs, _out):
        return op.Expand(rhs, op.Constant(value_ints=list(_out.shape)))
