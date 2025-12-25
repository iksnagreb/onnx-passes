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
# TODO: Not used anymore...
ALWAYS_FOLD_OPS = {
    "Transpose", "Constant", "ConstantOfShape", "Reshape", "Not", "Split"
}

# Disables all size limits ofr constant-folding
NO_LIMITS = {
    "input_size_limit": np.inf, "output_size_limit": np.inf
}


# TODO: Come up with more clever folding strategies, for now this tries to fold
#  everything, which is probably fine with MatMul scale factor extraction?
def _should_fold(_: ir.Node):
    return True


# Performs constant folding on the entire model graph
@passes.verify.tolerance
@passes.register("fold-constants")
class FoldConstants(Transformation):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Configure constant folding behavior: Size limits of constant foldable
        # tensors - disable all size limits by default
        kwargs = self.config.setdefault(
            "fold_constants", {**NO_LIMITS, "onnx_shape_inference": True}
        )
        # Run in-place constant - yields PassResult
        modified = fold_constants(model, should_fold=_should_fold,
                                  **kwargs).modified
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
class FoldConstantSize(Transformation, RewriteRulePass):
    def pattern(self, op, x):
        return op.Size(x)

    def check(self, _, x: ir.Value):
        return x.shape and all(isinstance(dim, int) for dim in x.shape)

    def rewrite(self, op, x):
        return op.Constant(value_int=int(np.prod(x.shape)))


# Registers custom constant folding for operators
from onnxscript.optimizer._constant_folding import register  # noqa: Protected


@register("Split")
def _fold_constants_split(node: ir.Node, op, _):
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


# Domain used by custom operators implemented with this library
from onnx_passes.ops import DOMAIN as CUSTOM_DOMAIN


@register("Im2Col", domain=CUSTOM_DOMAIN)
def _fold_constants_im2col(node: ir.Node, op, _):
    # Skip if there are not exactly two inputs as required by our custom-op
    # specification (proper input + pre-computed access pattern)
    if len(node.inputs) != 2:
        return None

    # Constant folding requires both of these inputs to be constants, otherwise
    # there is nothing to fold...
    if (x := ir.convenience.get_const_tensor(node.inputs[0])) is None:
        return None

    if (indices := ir.convenience.get_const_tensor(node.inputs[1])) is None:
        return None

    # From Im2Col operator definition, see onnx_passes.ops.im2col, slightly
    # adjusted from ONNX to NumPy behavior
    return op.Constant(
        value=ir.tensor(x.numpy().reshape(x.shape[0], -1)[:, indices.numpy()])
    )


# Use ONNX Script implementation of inverse Swish and Silu in eager mode
# evaluation, i.e., executing the Python/Numpy, during constant folding
from onnx_passes.ops.inverse_swish import InverseSwish, InverseSilu


@register("InverseSwish", domain=CUSTOM_DOMAIN)
def _fold_constants_inverse_swish(node: ir.Node, op, _):
    # Skip if there is not exactly one input (Swish and its inverse are unary)
    if len(node.inputs) != 1:
        return None

    # Constant folding requires the input to be constants, otherwise there is
    # nothing to fold...
    if (x := ir.convenience.get_const_tensor(node.inputs[0])) is None:
        return None

    # Default Swish alpha is 1.0, according to ONNX operators reference:
    #   https://onnx.ai/onnx/operators/onnx__Swish.html
    if (alpha := node.attributes.get("alpha")) is None:
        alpha = ir.Attr("alpha", ir.AttributeType.FLOAT, 1.0)

    # Default InverseSwish k is 0, i.e., the principal branch
    if (k := node.attributes.get("k")) is None:
        k = ir.Attr("k", ir.AttributeType.INT, 0)

    # Use the eager mode evaluation to generate the folded constant node
    return op.Constant(value=ir.tensor(
        InverseSwish(x.numpy(), k=k.as_int(), alpha=alpha.as_int())
    ))


@register("InverseSilu", domain=CUSTOM_DOMAIN)
def _fold_constants_inverse_silu(node: ir.Node, op, _):
    # Skip if there is not exactly one input (Swish and its inverse are unary)
    if len(node.inputs) != 1:
        return None

    # Constant folding requires the input to be constants, otherwise there is
    # nothing to fold...
    if (x := ir.convenience.get_const_tensor(node.inputs[0])) is None:
        return None

    # Default InverseSilu k is 0, i.e., the principal branch
    if (k := node.attributes.get("k")) is None:
        k = ir.Attr("k", ir.AttributeType.INT, 0)

    # Use the eager mode evaluation to generate the folded constant node
    return op.Constant(value=ir.tensor(InverseSilu(x.numpy(), k=k.as_int())))


# Use ONNX Script implementation of ArgSort in eager mode evaluation, i.e.,
# executing the Python/Numpy, during constant folding
from onnx_passes.ops.argsort import ArgSort


@register("ArgSort", domain=CUSTOM_DOMAIN)
def _fold_constants_argsort(node: ir.Node, op, _):
    # Skip if there is not exactly one input (ArgSort does not accept further
    # inputs)
    if len(node.inputs) != 1:
        return None

    # Constant folding requires the input to be constants, otherwise there is
    # nothing to fold...
    if (x := ir.convenience.get_const_tensor(node.inputs[0])) is None:
        return None

    # Default ArgSort axis is -1, i.e., sort along the last axis
    if (axis := node.attributes.get("axis")) is None:
        axis = ir.Attr("axis", ir.AttributeType.INT, -1)

    # Use the eager mode evaluation to generate the folded constant node
    return op.Constant(value=ir.tensor(ArgSort(x.numpy(), axis=axis.as_int())))


# Use ONNX Runtime as evaluator for some constant folding implementations
# instead of the default ONNX Reference evaluator
from onnxscript.evaluator import ORTEvaluator
from onnx.onnx_cpp2py_export.defs import get_schema

_ort_evaluator = ORTEvaluator()


def _ort_evaluate(node: ir.Node, *args, **kwargs):
    # If the node itself does not specify the operator version, use the version
    # from the graphs opset imports
    version = node.version

    if version is None:
        version = node.graph.opset_imports[node.domain]

    # Evaluate th node with all args interpreted as inputs and keyword arguments
    # interpreted as attributes
    return _ort_evaluator.eval(
        get_schema(node.op_type, version, node.domain), [*args], {**kwargs}
    )


# Default constant folding of GatherElements uses the ONNX reference evaluator
# which implements gathering in NumPy via np.choose which seems to have issues
# with large inputs (or rather inputs with many elements along the axis).
@register("GatherElements")
def _fold_constants_gather_elements(node: ir.Node, op, _):
    # Skip if there is not exactly one input (Gather does not accept fewer or
    # further inputs)
    if len(node.inputs) != 2:
        return None

    # Constant folding requires both inputs to be constants, otherwise there is
    # nothing to fold...
    if (x := ir.convenience.get_const_tensor(node.inputs[0])) is None:
        return None

    # Constant folding requires both inputs to be constants, otherwise there is
    # nothing to fold...
    if (indices := ir.convenience.get_const_tensor(node.inputs[1])) is None:
        return None

    # Default GatherElements axis is 0, according to ONNX operators reference:
    #   https://onnx.ai/onnx/operators/onnx__GatherElements.html
    if (axis := node.attributes.get("axis")) is None:
        axis = ir.Attr("axis", ir.AttributeType.INT, 0)

    # Use the ONNX Runtime evaluator to execute the node instead of the default
    # reference evaluator
    return op.Constant(value=ir.tensor(
        _ort_evaluate(node, x.numpy(), indices.numpy(), axis=axis.value).value
    ))
