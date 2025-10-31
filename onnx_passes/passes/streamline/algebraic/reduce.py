# ir.Value
import onnx_ir as ir

# All algebraic passes are transformations derived from pattern-based rewrite
# rules
from onnx_passes.passes.base import Transformation, RewriteRulePass

# Checking ir.Value for being constants
from onnx_passes.passes.util import is_constant, is_scalar, unbroadcast

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# NumPy used during match condition checks to operate on shapes and tensors
import numpy as np


# Moves elementwise additions past sum-reduction exploiting commutativity and
# associativity of addition and broadcasting of elementwise operations.
@passes.verify.tolerance
@passes.register("algebraic")
class MoveAddPastReduceSum(Transformation, RewriteRulePass):
    def pattern(self, op, x, y, axes, keepdims):
        return op.ReduceSum(op.Add(x, y), axes, keepdims=keepdims)

    # Avoid increasing the number of non constant-foldable reductions
    def check(self, op, x, y, axes, keepdims):
        return is_constant(x) or is_constant(y)

    def rewrite(self, op, x, y, axes, keepdims):
        return op.Add(
            op.ReduceSum(
                op.Expand(x, op.Shape(op.Add(x, y))), axes, keepdims=keepdims
            ),
            op.ReduceSum(
                op.Expand(y, op.Shape(op.Add(x, y))), axes, keepdims=keepdims
            )
        )


# Checks the condition for reordering certain elementwise-reduce combinations
# which require at least one input to be the same for each slice along the
# reduction axes.
#
# For static shapes, we can check if the axes are explicitly or implicitly 1 via
# the following trick: An auxiliary tensor of the same shape is broadcast and
# immediately unbroadcast while keeping the rank which yields a shape compatible
# with the axes and encodes repetitions as empty, i.e., size 1, axes.
def _is_broadcast_along_axes(shape, other, axes):
    # Generate auxiliary of the same shape without redundancies in the initial
    # data, i.e., this is maximally unbroadcast and any repetitions are due to
    # broadcasting according to static shape information
    auxiliary = np.arange(np.prod(shape)).reshape(shape)

    # Mimic the broadcasting of the elementwise operation to have compatible
    # shapes, but redundancies
    auxiliary = np.broadcast_to(auxiliary, np.broadcast_shapes(other, shape))

    # Unbroadcasting keeping the rank, now all reduction axes should have
    # corresponding auxiliary axes
    auxiliary = unbroadcast(auxiliary, squeeze=False)

    # If the auxiliary has empty axes for all reduction axes, the actual tensor
    # will repeat the value along these axes.
    return np.all(np.asarray(auxiliary.shape)[axes] == 1)


# Moves elementwise multiplications past sum-reduction exploiting distributivity
# of multiplication over addition, given one of the scales is the same for each
# slice along the reduction axes.
@passes.verify.tolerance
@passes.register("algebraic")
class MoveMulPastReduceSum(Transformation, RewriteRulePass):
    # Commutativity applies to the multiplication so we can formulate match
    # conditions and rewrite patterns for one input (y) only but make the rules
    # apply symmetrically to both inputs.
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y, axes, keepdims):
        return op.ReduceSum(op.Mul(x, y), axes, keepdims=keepdims)

    def check(self, op, x, y, axes, keepdims):
        # The general case requires constant axes to check whether the same
        # value is repeated (broadcast) along these axes
        if (axes := ir.convenience.get_const_tensor(axes)) is not None:
            # This requires static shape information for both inputs, but no
            # actual values for any of the inputs
            if y.shape is not None and y.shape.is_static():
                if x.shape is not None and x.shape.is_static():
                    if _is_broadcast_along_axes(y.shape, x.shape, axes.numpy()):
                        # Avoid increasing the number of non constant-foldable
                        # reductions for now
                        # TODO: Could be relaxed once there is elimination of
                        #  identity-reduce, i.e., reductions along empty axes.
                        return is_constant(y)

            # Note: There could be an alternative formulation if y is a constant
            # tensor, but this still requires a static shape for the other input
            # x in most cases. As y being a constant usually implies a static
            # shape, this seems equivalent to the above...
            # TODO: This assumes y is maximally unbroadcast when applying this
            #  transformation, otherwise we miss true redundancies not captured
            #  by static shape information

        # Trivial case: Input y is a scalar, which is the same along all axes
        # and compatible with any shape, also no reduction/broadcasting needed
        return is_scalar(y)

    def rewrite(self, op, x, y, axes, keepdims):
        return op.Mul(
            op.ReduceSum(
                op.Expand(x, op.Shape(op.Mul(x, y))), axes, keepdims=keepdims
            ),
            # This might look weird, but the match condition ensures that all y
            # are the same along the reduction axes, so a min-reduction will
            # just get rid of these axes to keep the shape as expected.
            # TODO: Alternatively a Slice could be inserted, but as long as we
            #  require y to be constant it does not really matter...
            op.ReduceMin(
                op.Expand(y, op.Shape(op.Mul(x, y))), axes, keepdims=keepdims
            ),
        )


# Moves elementwise additions past max-reduction exploiting distributivity
# of addition over maximum, given one of the summands is the same for each slice
# along the reduction axes.
@passes.verify.tolerance
@passes.register("algebraic")
class MoveAddPastReduceMax(Transformation, RewriteRulePass):
    # Commutativity applies to the addition so we can formulate match conditions
    # and rewrite patterns for one input (y) only but make the rules apply
    # symmetrically to both inputs.
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y, axes, keepdims):
        return op.ReduceMax(op.Add(x, y), axes, keepdims=keepdims)

    def check(self, op, x, y, axes, keepdims):
        # The general case requires constant axes to check whether the same
        # value is repeated (broadcast) along these axes
        if (axes := ir.convenience.get_const_tensor(axes)) is not None:
            # This requires static shape information for both inputs, but no
            # actual values for any of the inputs
            if y.shape is not None and y.shape.is_static():
                if x.shape is not None and x.shape.is_static():
                    if _is_broadcast_along_axes(y.shape, x.shape, axes.numpy()):
                        # Avoid increasing the number of non constant-foldable
                        # reductions for now
                        # TODO: Could be relaxed once there is elimination of
                        #  identity-reduce, i.e., reductions along empty axes.
                        return is_constant(y)

            # Note: There could be an alternative formulation if y is a constant
            # tensor, but this still requires a static shape for the other input
            # x in most cases. As y being a constant usually implies a static
            # shape, this seems equivalent to the above...
            # TODO: This assumes y is maximally unbroadcast when applying this
            #  transformation, otherwise we miss true redundancies not captured
            #  by static shape information

        # Trivial case: Input y is a scalar, which is the same along all axes
        # and compatible with any shape, also no reduction/broadcasting needed
        return is_scalar(y)

    def rewrite(self, op, x, y, axes, keepdims):
        return op.Add(
            op.ReduceMax(
                op.Expand(x, op.Shape(op.Add(x, y))), axes, keepdims=keepdims
            ),
            op.ReduceMax(
                op.Expand(y, op.Shape(op.Add(x, y))), axes, keepdims=keepdims
            )
        )


# Moves elementwise additions past min-reduction exploiting distributivity
# of addition over minimum, given one of the summands is the same for each slice
# along the reduction axes.
@passes.verify.tolerance
@passes.register("algebraic")
class MoveAddPastReduceMin(Transformation, RewriteRulePass):
    # Commutativity applies to the addition so we can formulate match conditions
    # and rewrite patterns for one input (y) only but make the rules apply
    # symmetrically to both inputs.
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y, axes, keepdims):
        return op.ReduceMin(op.Add(x, y), axes, keepdims=keepdims)

    def check(self, op, x, y, axes, keepdims):
        # The general case requires constant axes to check whether the same
        # value is repeated (broadcast) along these axes
        if (axes := ir.convenience.get_const_tensor(axes)) is not None:
            # This requires static shape information for both inputs, but no
            # actual values for any of the inputs
            if y.shape is not None and y.shape.is_static():
                if x.shape is not None and x.shape.is_static():
                    if _is_broadcast_along_axes(y.shape, x.shape, axes.numpy()):
                        # Avoid increasing the number of non constant-foldable
                        # reductions for now
                        # TODO: Could be relaxed once there is elimination of
                        #  identity-reduce, i.e., reductions along empty axes.
                        return is_constant(y)

            # Note: There could be an alternative formulation if y is a constant
            # tensor, but this still requires a static shape for the other input
            # x in most cases. As y being a constant usually implies a static
            # shape, this seems equivalent to the above...
            # TODO: This assumes y is maximally unbroadcast when applying this
            #  transformation, otherwise we miss true redundancies not captured
            #  by static shape information

        # Trivial case: Input y is a scalar, which is the same along all axes
        # and compatible with any shape, also no reduction/broadcasting needed
        return is_scalar(y)

    def rewrite(self, op, x, y, axes, keepdims):
        return op.Add(
            op.ReduceMin(
                op.Expand(x, op.Shape(op.Add(x, y))), axes, keepdims=keepdims
            ),
            op.ReduceMin(
                op.Expand(y, op.Shape(op.Add(x, y))), axes, keepdims=keepdims
            )
        )


# Moves elementwise multiplication past max-reduction exploiting distributivity
# of multiplication over maximum for non-negative factors, given one of the
# factors is the same for each slice along the reduction axes.
@passes.verify.tolerance
@passes.register("algebraic")
class MoveMulPastReduceMax(Transformation, RewriteRulePass):
    # Commutativity applies to the multiplication so we can formulate match
    # conditions and rewrite patterns for one input (y) only but make the rules
    # apply symmetrically to both inputs.
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y, axes, keepdims):
        return op.ReduceMax(op.Mul(x, y), axes, keepdims=keepdims)

    def check(self, op, x, y, axes, keepdims):
        # The general case requires constant axes to check whether the same
        # value is repeated (broadcast) along these axes
        if (axes := ir.convenience.get_const_tensor(axes)) is not None:
            # This requires static shape information for both inputs, but no
            # actual values for any of the inputs
            if y.shape is not None and y.shape.is_static():
                if x.shape is not None and x.shape.is_static():
                    if _is_broadcast_along_axes(y.shape, x.shape, axes.numpy()):
                        # Avoid increasing the number of non constant-foldable
                        # reductions for now
                        # TODO: Could be relaxed once there is elimination of
                        #  identity-reduce, i.e., reductions along empty axes.
                        return is_constant(y)

            # Note: There could be an alternative formulation if y is a constant
            # tensor, but this still requires a static shape for the other input
            # x in most cases. As y being a constant usually implies a static
            # shape, this seems equivalent to the above...
            # TODO: This assumes y is maximally unbroadcast when applying this
            #  transformation, otherwise we miss true redundancies not captured
            #  by static shape information

        # Trivial case: Input y is a scalar, which is the same along all axes
        # and compatible with any shape, also no reduction/broadcasting needed
        return is_scalar(y)

    def rewrite(self, op, x, y, axes, keepdims):
        return op.Mul(
            # Negative factors flip maximum to minimum, which we have to apply
            # elementwise here, based on the sign of the moved factor y
            op.Where(
                # As the factor y does not change over the reduction axes, it
                # does not matter whether the min- or max-reduced y is used
                op.GreaterOrEqual(
                    op.ReduceMax(
                        op.Expand(y, op.Shape(op.Mul(x, y))), axes,
                        keepdims=keepdims
                    ),
                    op.CastLike(op.Constant(value_int=0), y)
                ),
                op.ReduceMax(
                    op.Expand(x, op.Shape(op.Mul(x, y))), axes,
                    keepdims=keepdims
                ),
                op.ReduceMin(
                    op.Expand(x, op.Shape(op.Mul(x, y))), axes,
                    keepdims=keepdims
                )
            ),
            # As the factor y does not change over the reduction axes, it does
            # not matter whether the min- or max-reduced y is used
            op.ReduceMax(
                op.Expand(y, op.Shape(op.Mul(x, y))), axes, keepdims=keepdims
            )
        )


# Moves elementwise multiplication past min-reduction exploiting distributivity
# of multiplication over minimum for non-negative factors, given one of the
# factors is the same for each slice along the reduction axes.
@passes.verify.tolerance
@passes.register("algebraic")
class MoveMulPastReduceMin(Transformation, RewriteRulePass):
    # Commutativity applies to the multiplication so we can formulate match
    # conditions and rewrite patterns for one input (y) only but make the rules
    # apply symmetrically to both inputs.
    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, y, axes, keepdims):
        return op.ReduceMin(op.Mul(x, y), axes, keepdims=keepdims)

    def check(self, op, x, y, axes, keepdims):
        # The general case requires constant axes to check whether the same
        # value is repeated (broadcast) along these axes
        if (axes := ir.convenience.get_const_tensor(axes)) is not None:
            # This requires static shape information for both inputs, but no
            # actual values for any of the inputs
            if y.shape is not None and y.shape.is_static():
                if x.shape is not None and x.shape.is_static():
                    if _is_broadcast_along_axes(y.shape, x.shape, axes.numpy()):
                        # Avoid increasing the number of non constant-foldable
                        # reductions for now
                        # TODO: Could be relaxed once there is elimination of
                        #  identity-reduce, i.e., reductions along empty axes.
                        return is_constant(y)

            # Note: There could be an alternative formulation if y is a constant
            # tensor, but this still requires a static shape for the other input
            # x in most cases. As y being a constant usually implies a static
            # shape, this seems equivalent to the above...
            # TODO: This assumes y is maximally unbroadcast when applying this
            #  transformation, otherwise we miss true redundancies not captured
            #  by static shape information

        # Trivial case: Input y is a scalar, which is the same along all axes
        # and compatible with any shape, also no reduction/broadcasting needed
        return is_scalar(y)

    def rewrite(self, op, x, y, axes, keepdims):
        return op.Mul(
            # Negative factors flip minimum to maximum, which we have to apply
            # elementwise here, based on the sign of the moved factor y
            op.Where(
                # As the factor y does not change over the reduction axes, it
                # does not matter whether the min- or max-reduced y is used
                op.GreaterOrEqual(
                    op.ReduceMin(
                        op.Expand(y, op.Shape(op.Mul(x, y))), axes,
                        keepdims=keepdims
                    ),
                    op.CastLike(op.Constant(value_int=0), y)
                ),
                op.ReduceMin(
                    op.Expand(x, op.Shape(op.Mul(x, y))), axes,
                    keepdims=keepdims
                ),
                op.ReduceMax(
                    op.Expand(x, op.Shape(op.Mul(x, y))), axes,
                    keepdims=keepdims
                )
            ),
            # As the factor y does not change over the reduction axes, it does
            # not matter whether the min- or max-reduced y is used
            op.ReduceMin(
                op.Expand(y, op.Shape(op.Mul(x, y))), axes, keepdims=keepdims
            )
        )
