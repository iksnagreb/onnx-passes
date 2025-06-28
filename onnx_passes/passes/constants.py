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

# NumPy used to operate on shapes and constant tensors
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


# Replaces Shape operators with Constant operators of the input tensor shape to
# enable constant folding of shape calculations - dynamic shapes (or missing
# shapes) are not supported
@passes.verify.equality
@passes.register("fold-constants")
class FoldConstantShape(Transformation, RewriteRulePass):
    def pattern(self, op, x):
        return op.Shape(x)

    def check(self, _, x: ir.Value):
        return x.shape and all(isinstance(dim, int) for dim in x.shape)

    def rewrite(self, op, x):
        return op.Constant(value_ints=list(x.shape))


# Replaces Size operators with Constant operators of the input tensor size to
# enable constant folding of shape calculations - dynamic shapes (or missing
# shapes) are not supported
@passes.verify.equality
@passes.register("fold-constants")
@passes.register()
class FoldConstantSize(Transformation, RewriteRulePass):
    def pattern(self, op, x):
        return op.Size(x)

    def check(self, _, x: ir.Value):
        return x.shape and all(isinstance(dim, int) for dim in x.shape)

    def rewrite(self, op, x):
        return op.Constant(value_int=int(np.prod(x.shape)))
