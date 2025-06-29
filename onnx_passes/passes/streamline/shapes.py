# ir.Value
import onnx_ir as ir

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# All algebraic passes are transformations derived from pattern-based rewrite
# rules
from onnx_passes.passes.base import Transformation, RewriteRulePass

# NumPy used for calculations on shapes and constant tensors in rewrites and
# match conditions
import numpy as np


# ==============================================================================
# The following deals with shape-related operations in a particular way: Shape
# propagation, operator fusion and constant elimination is formulated in terms
# of normalized Reshape operations.
#
# To support these optimization for any other type of shape-related operations,
# such as Flatten, Squeeze, Unsqueeze or non-default Reshape, these are first
# converted and normalized to Reshape operations.
#
# For some of these transformations two "styles" are implemented below: a static
# shape calculation version which needs input shapes and axes to be constants
# and immediately calculates a constant output shape, and a "dynamic" shape
# calculation version which inserts the ONNX equivalent of these calculations
# into the graph. If both styles are available, the static version is forced for
# now, as the dynamic version produces rather verbose output, which, even though
# it seems to be perfectly constant-foldable, tends to be hard to debug...
#
# TODO: Consider switching all transformations to dynamic shape calculations
#  once this part of streamlining is finished.
# ==============================================================================

# Expresses Flatten operations by Reshape operations allows to express all shape
# propagation and simplification in terms of Reshape
@passes.verify.equality
@passes.register("streamline-shapes")
class ConvertFlattenToReshape(Transformation, RewriteRulePass):
    def pattern(self, op, x):
        return op.Flatten(x, _outputs=["y"])

    def check(self, op, x, y):
        return x.shape and all(isinstance(dim, int) for dim in x.shape)

    def rewrite(self, op, x, y):
        # Default axis according to ONNX operators reference documentation:
        #   https://onnx.ai/onnx/operators/onnx__Flatten.html
        if not (axis := y.producer().attributes.get("axis", None)):
            axis = ir.Attr("axis", ir.AttributeType.INT, 1)

        # According to ONNX reference always reshapes into a 2D matrix with
        # flattened dimensions up to and starting from axis
        shape = [
            int(np.prod(x.shape[:axis.as_int()])),
            int(np.prod(x.shape[axis.as_int():]))
        ]

        # Insert constant reshape representation of the Flatten operation
        return op.Reshape(x, op.Constant(value_ints=shape))

        # ======================================================================
        # The following is the "dynamic" shape equivalent not depending on the
        # input shape being a constant.
        #
        # The dynamic version is still constant-foldable resulting in the
        # equivalent output as the static version above.
        # ======================================================================

        # # Start and end for slicing the input shape into two sections
        # # controlled by the axis attribute
        # start = op.Constant(value_ints=[0])
        # axis = op.Constant(value_ints=[axis.as_int()])
        #
        # # All elements must be used by the output: the first output dimension
        # # covers all dimensions up to (excluding) the axis, while the second
        # # dimension covers what is remaining, note that the empty product is
        # # defined to be one.
        # dim0 = op.ReduceProd(op.Slice(op.Shape(x), start, axis))
        # dim1 = op.Div(op.ReduceProd(op.Shape(x)), dim0)
        #
        # # Combine the two dimensions along the single first axis forming a
        # # 2-dimensional shape
        # return op.Reshape(x, op.Concat(dim0, dim1, axis=0))


# Infers squeeze axes from a default, missing axes input which implicitly means
# squeezing all single-dimensional entries from the input shape.
@passes.verify.equality
@passes.register("streamline-shapes")
class InferSqueezeAxes(Transformation, RewriteRulePass):
    def pattern(self, op, x):
        return op.Squeeze(x)

    def rewrite(self, op, x):
        # ======================================================================
        # TODO: Come up with some static shape equivalent of this as well, even
        #  though there is probably no practical benefit...
        # ======================================================================

        # Find all single-dimensional entries from the shape, inserting
        # potentially dynamic shape calculations
        axes = op.NonZero(op.Equal(op.Shape(x), op.Constant(value_int=1)))
        # Assemble squeeze operator with explicit axes input after getting rid
        # of ome extra dimension inserted by op.NonZero
        return op.Squeeze(x, op.Reshape(axes, op.Constant(value_ints=[-1])))


# Expresses Squeeze operations by Reshape operations allows to express all shape
# propagation and simplification in terms of Reshape
@passes.verify.equality
@passes.register("streamline-shapes")
class ConvertSqueezeToReshape(Transformation, RewriteRulePass):
    def pattern(self, op, x, axes):
        return op.Squeeze(x, axes)

    def check(self, op, x, axes):
        if ir.convenience.get_const_tensor(axes) is not None:
            return x.shape and all(isinstance(dim, int) for dim in x.shape)
        return False

    def rewrite(self, op, x, axes):
        # Already made sure to have constant axes via the match condition, thus
        # converting this to NumPy format is safe
        axes = ir.convenience.get_const_tensor(axes).numpy()
        # Derive the output shape by deleting the axes from the input shape,
        # assuming the graph to be in a valid state, i.e., never deleting non
        # single-dimensional entries from the shape.
        shape = np.delete(np.asarray(x.shape), axes)
        # Rewrite the squeeze as a constant reshape operation - op.Constant
        # needs a list, not NumPy array...
        return op.Reshape(x, op.Constant(value_ints=shape.tolist()))

        # ======================================================================
        # The following is the "dynamic" shape equivalent not depending on the
        # input shape or the axes being a constant.
        #
        # The dynamic version is still constant-foldable resulting in the
        # equivalent output as the static version above.
        # ======================================================================

        # # Mark axes selected to be squeezed by negative sizes (these cannot
        # # appear as the output of op.Shape by default)
        # shape = op.ScatterElements(
        #     op.Shape(x), axes, op.Expand(
        #         op.Constant(value_int=-1), op.Shape(axes)
        #     )
        # )
        #
        # # Generate indices of all entries from the input shape which are not
        # # marked by -1, i.e., those entries to keep
        # # Note: there seems to be no "if i not in axes" ONNX equivalent
        # keep = op.NonZero(op.Not(op.Equal(shape, op.Constant(value_int=-1))))
        #
        # # Select all entries from the input shape to keep after getting rid of
        # # some extra dimension inserted by op.NonZero
        # shape = op.GatherElements(
        #     op.Shape(x), op.Reshape(keep, op.Constant(value_ints=[-1]))
        # )
        #
        # # Use the (dynamic) shape calculation as second input to the reshape
        # # operation finally replacing the squeeze
        # return op.Reshape(x, shape)


# Expresses Unsqueeze operations by Reshape operations allows to express all
# shape propagation and simplification in terms of Reshape
@passes.verify.equality
@passes.register("streamline-shapes")
class ConvertUnsqueezeToReshape(Transformation, RewriteRulePass):
    def pattern(self, op, x, axes):
        return op.Unsqueeze(x, axes)

    def check(self, op, x, axes):
        if ir.convenience.get_const_tensor(axes) is not None:
            return x.shape and all(isinstance(dim, int) for dim in x.shape)
        return False

    def rewrite(self, op, x, axes):
        # Already made sure to have constant axes via the match condition, thus
        # converting this to NumPy format is safe
        axes = ir.convenience.get_const_tensor(axes).numpy()
        # Derive the output shape by inserting single-dimensional entries at the
        # axes into the input shape, assuming the graph to be in a valid state,
        # i.e., inserting duplicate non single-dimensional entries.
        shape = np.expand_dims(np.ones(x.shape), list(axes)).shape
        # Rewrite the unsqueeze as a constant reshape operation - op.Constant
        # needs a list, not NumPy array...
        return op.Reshape(x, op.Constant(value_ints=shape))

        # ======================================================================
        # The following is the "dynamic" shape equivalent not depending on the
        # input shape or the axes being a constant.
        #
        # The dynamic version is still constant-foldable resulting in the
        # equivalent output as the static version above.
        # ======================================================================

        # # All zero and all one tensors covering the axes used for repeatedly
        # # updating the indices and shape calculated below
        # _0 = op.Expand(op.Constant(value_int=0), op.Shape(axes))
        # _1 = op.Expand(op.Constant(value_int=1), op.Shape(axes))
        #
        # # The rank of the unsqueezed output: Old rank + inserted dimensions
        # rank = op.Add(op.Size(op.Shape(x)), op.Size(axes))
        #
        # # Start operating on a sequence of indices mapping from new to old
        # # dimensions: Seed mapping to 1-based indexing...
        # indices = op.ConstantOfShape(
        #     op.Reshape(rank, op.Constant(value_ints=[-1])),
        #     value=ir.tensor([1])
        # )
        #
        # # Update the index mapping by (1) skipping the unsqueezed dimensions,
        # # (2) cumulatively adding up the input dimensions and, (3) subtracting
        # # one to move to a zero-based indexing
        # indices = op.Sub(
        #     op.CumSum(
        #         op.ScatterElements(indices, axes, _0),
        #         op.Constant(value_int=0)
        #     ),
        #     op.Constant(value_int=1)
        # )
        #
        # # Derive the output shape by (1) collecting input dimensions according
        # # to the index mapping and, (2) updating the shape by setting all
        # # unsqueezed dimension to 1
        # shape = op.ScatterElements(
        #     op.GatherElements(op.Shape(x), indices), axes, _1
        # )
        #
        # # Use the (dynamic) shape calculation as second input to the reshape
        # # operation finally replacing the unsqueeze
        # return op.Reshape(x, shape)


# Fuses two consecutive reshape operations into a single reshape, i.e. producing
# the output shape of the second reshape, effectively eliminating the first
@passes.verify.equality
@passes.register("streamline-shapes")
class FuseReshape(Transformation, RewriteRulePass):
    def pattern(self, op, x, shape1, shape2):
        return op.Reshape(op.Reshape(x, shape1), shape2, _outputs=["y"])

    def rewrite(self, op, x, shape1, shape2, y):
        # Default allowzero according to ONNX operators reference documentation:
        #   https://onnx.ai/onnx/operators/onnx__Reshape.html
        if not (allowzero := y.producer().attributes.get("allowzero", None)):
            allowzero = ir.Attr("allowzero", ir.AttributeType.INT, 0)

        # ======================================================================
        # TODO: Come up with some static shape equivalent of this as well, even
        #  though there is probably no practical benefit...
        # ======================================================================

        # Start by assuming the shape of the second reshape to fully determine
        # the final output shape, which is almost always the case
        shape = shape2

        # Turn allowzero=0 pass-through dimensions of the second reshape into
        # explicit dimensions inferred from the shape of the first reshape
        if allowzero is None or allowzero.as_int() == 0:
            # Find indices of dimensions to be passed through from the shape of
            # the first reshape, i.e., those where the second shape has zeros
            i = op.Reshape(
                op.NonZero(op.Equal(shape2, op.Constant(value_int=0))),
                op.Constant(value_ints=[-1])
            )

            # Update the output shape with pass-through entries gathered from
            # the intermediate shape
            shape = op.ScatterElements(shape2, i, op.GatherElements(shape1, i))

        # Fused reshape keeping the allowzero attribute of the second reshape
        return op.Reshape(x, shape, allowzero=allowzero.as_int())

# TODO: Implement reshape propagation through elementwise and MatMul (if
#  applicable) segments of the graph
