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

# Operators which should always be constant folded
ALWAYS_FOLD_OPS = {
    "always_fold_ops": {
        "Transpose", "Constant", "ConstantOfShape", "Reshape", "Not"
    }
}


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
        modified = fold_constants(model, **ALWAYS_FOLD_OPS).modified
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


# Registers custom constant folding for operators
from onnxscript.optimizer._constant_folding import register  # noqa: Protected


# Replaces Split operators with all constant inputs by a list of Constant
# operators
@register("Split")
def split(node: ir.Node, op, _):
    # Replace single output split by Identity(x)
    if len(node.outputs) == 1:
        return op.Identity(node.inputs[0])

    # Skip non-constant inputs
    if (x := ir.convenience.get_const_tensor(node.inputs[0])) is None:
        return None

    _split = None

    # Option A: Sizes per split
    if len(node.inputs) == 2:
        # Skip non-constant splits
        if (_split := ir.convenience.get_const_tensor(node.inputs[1])) is None:
            return None
        # Numpy expects splits as starting indices for each section
        _split = np.cumsum(_split.numpy()[:-1])

    # Option B: Number of (even) splits
    if (num_outputs := node.attributes.get("num_outputs")) is not None:
        # Numpy accepts single integer of (even) splits as well
        _split = num_outputs.as_int()

    # Hm, something must be terribly wrong...
    if _split is None:
        return None

    # Default split axis is 0, according to ONNX operators reference:
    #   https://onnx.ai/onnx/operators/onnx__Split.html
    if (axis := node.attributes.get("axis")) is None:
        axis = ir.Attr("axis", ir.AttributeType.INT, 0)

    # Split constant tensor and wrap a list of Constant operators
    splits = np.split(x.numpy(), _split, axis.as_int())
    return [op.Constant(value=ir.tensor(x)) for x in splits]
