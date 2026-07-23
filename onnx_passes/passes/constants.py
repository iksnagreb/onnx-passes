# ONNX intermediate representation
import onnx_ir as ir
# NumPy used to operate on shapes and constant tensors
import numpy as np

# Common cleanup passes already implemented in ONNX IR, used here without any
# custom infrastructure.
import onnx_ir.passes.common

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# Derive Transformations (allowed to modify the graph) from pattern-based
# rewrite rules
from onnx_passes.passes.base import Transformation, RewriteRulePass

# Custom operators implemented via ONNX Script functions
from onnx_passes.ops import link_ops

# Use the ONNX reference evaluator to evaluate nodes for constant folding
from onnx.reference import ReferenceEvaluator

# Imperative builder for constructing ONNX IR graphs
from onnxscript import GraphBuilder


def _cleanup(model: ir.Model, inplace: bool = True) -> ir.passes.PassResult:
    """Performs basic cleanup of the model graph."""

    # Optionally cleanup a deep copy of the model and keep the original
    # model unmodified. Deep copy to not entangle the metadata stores.
    if not inplace:
        model = model.clone(deep_copy=True)

    # Run a sequence of ONNX IR cleanup passes, starting with the graph in
    # topological order (which is itself required in a cleaned-up) to not miss
    # any disconnected, unused nodes.
    cleanup = ir.passes.PassManager([
        ir.passes.common.TopologicalSortPass(),
        ir.passes.common.RemoveUnusedNodesPass(),
        ir.passes.common.RemoveUnusedFunctionsPass(),
        ir.passes.common.RemoveUnusedOpsetsPass(),
        ir.passes.common.LiftConstantsToInitializersPass(),
        ir.passes.common.RemoveInitializersFromInputsPass(),
        ir.passes.common.DeduplicateInitializersPass(),
        ir.passes.common.ShapeInferencePass()
    ])

    return cleanup(model)


# Backlists operator types from evaluation-based constant folding
BLACKLIST: set[str] = {
    "Constant", "GatherElements"
}


def _fold_constants(model: ir.Model,
                    inplace: bool = True) -> ir.passes.PassResult:
    """Folds all constants in the model graph by evaluating the nodes."""

    # Optionally constant-fold a deep copy of the model and keep the original
    # model unmodified. Deep copy to not entangle the metadata stores.
    if not inplace:
        model = model.clone(deep_copy=True)

    # Link all missing custom operator functions from out domains into the model
    # graph
    model = link_ops(model)

    # Convert all functions currently linked into the model to proto
    # representation to make them available to the reference evaluator
    functions = [ir.to_proto(f) for f in model.functions.values()]

    # Bring the graph into topological order to evaluate all nodes before any of
    # their consumers to fold all constants in one pass
    model.graph.sort()

    # Graph builder for inserting replacement constant ops into the existing
    # graph
    builder = GraphBuilder(model.graph)
    op = builder.op

    # Keep track of whether the model is modified, i.e., at least one constant
    # is folded
    modified = False

    # Constant folding of all nodes in the graph in recursive topological order
    for node in ir.traversal.RecursiveGraphIterator(model.graph):
        # Folding 'Constant' operators to be replaced by 'Constant' operators
        # would iterate forever. Others might be black-listed for reasons...
        if node.op_type in BLACKLIST:
            continue

        # Collect all available constant inputs to the node as the execution
        # context mapping input name to numpy array
        context = {}

        for x in node.inputs:
            if x is not None:
                if (value := ir.convenience.get_const_tensor(x)) is not None:
                    context[x.name] = value.numpy()

        # Use the ONNX reference evaluator to execute the node on the context
        # collecting named outputs into the context as well
        try:
            session = ReferenceEvaluator(ir.to_proto(node), functions=functions)
            context = session.run(None, context, intermediate=True)

            # Replace each output of the node by the constant from the context,
            # rewiring all consumers, including the graph outputs.
            for y in node.outputs:
                if (name := y.name) is not None:
                    ir.convenience.replace_all_uses_with(
                        y, op.Constant(value=ir.tensor(context[name])), True
                    )
                    modified = True
        except RuntimeError:
            pass

    # Cleanup the graph removing now disconnected nodes
    result = _cleanup(model, inplace=True)
    return ir.passes.PassResult(result.model, result.modified or modified)


@passes.verify.equality
@passes.register("fold-constants")
class FoldConstantCastLike(Transformation, RewriteRulePass):
    """Folds CastLike into Cast if the target dtype is known."""

    def pattern(self, op, x, target):
        return op.CastLike(x, target)

    def check(self, op, x, target):
        return target.dtype is not None

    def rewrite(self, op, x, target):
        return op.Cast(x, to=target.dtype.value)


@passes.verify.equality
@passes.register("fold-constants")
class FoldConstantShape(Transformation, RewriteRulePass):
    """Folds Shape operators if the input shape is known."""

    def pattern(self, op, x):
        return op.Shape(x)

    def check(self, _, x: ir.Value):
        return x.shape is not None and x.shape.is_static()  # noqa: Never None

    def rewrite(self, op, x):
        return op.Constant(value_ints=list(x.shape))


@passes.verify.equality
@passes.register("fold-constants")
class FoldConstantSize(Transformation, RewriteRulePass):
    """Folds Size operators if the input shape is known."""

    def pattern(self, op, x):
        return op.Size(x)

    def check(self, _, x: ir.Value):
        return x.shape is not None and x.shape.is_static()  # noqa: Never None

    def rewrite(self, op, x):
        return op.Constant(value_int=int(np.prod(x.shape)))


@passes.verify.equality
@passes.register("fold-constants")
class FoldConstantGatherElements(Transformation, RewriteRulePass):
    """Folds GatherElements with both inputs constant.

    Note: This addresses some issue with the ONNX reference evaluator used for
    constant folding which implements gathering in NumPy via np.choose which
    seems to have issues with large inputs (or rather inputs with many elements
    along the axis).
    """

    def pattern(self, op, x, indices):
        return op.GatherElements(x, indices, _outputs=["y"])

    def check(self, op, x, indices, y):
        if ir.convenience.get_const_tensor(x) is not None:
            if ir.convenience.get_const_tensor(indices) is not None:
                return True
        return False

    def rewrite(self, op, x, indices, y):
        # Default axis is 0, according to ONNX operators reference:
        #   https://onnx.ai/onnx/operators/onnx__GatherElements.html
        if (axis := y.producer().attributes.get("axis")) is None:
            axis = ir.Attr("axis", ir.AttributeType.INT, 0)

        # Extract tensors and attributes as python/numpy objects
        axis = axis.as_int()

        x = ir.convenience.get_const_tensor(x).numpy()  # noqa: Never None
        indices = ir.convenience.get_const_tensor(indices).numpy()  # noqa: ...

        # Rearrange and flatten the tensors such that all indexing logic applies
        # to the final axis
        x_swapped = x.swapaxes(-1, axis)
        x = x_swapped.reshape(-1, x.shape[axis])
        indices = indices.swapaxes(-1, axis).reshape(-1, indices.shape[axis])

        # Output tensor shape matches the input tensor shape
        y = np.empty_like(x)

        # Gather logic in flattened final axis form
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                y[i][j] = x[i][indices[i][j]]

        # Restore the original shape and axis order
        y = y.reshape(x_swapped.shape).swapaxes(-1, axis)

        # Constant tensor replacement pattern
        return op.Constant(value=ir.tensor(y))


@passes.verify.tolerance
@passes.register("fold-constants")
class FoldConstants(Transformation):
    """Applies constant folding to the model."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return _fold_constants(model, inplace=True)
