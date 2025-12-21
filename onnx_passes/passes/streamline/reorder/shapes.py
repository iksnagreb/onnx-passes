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
@passes.register("reorder")
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
@passes.register("reorder")
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
@passes.register("reorder")
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
@passes.register("reorder")
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
@passes.register("reorder")
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


# Makes the implicit default attribute allowzero=0 explicit (among other things,
# this is assumed by threshold fusion)
@passes.register("reorder")
class ExplicitAllowzeroReshape(Transformation, RewriteRulePass):
    def pattern(self, op, x, shape):
        return op.Reshape(x, shape, _outputs=["y"])

    def check(self, op, x, shape, y):
        # Default allowzero according to ONNX operators reference documentation:
        #   https://onnx.ai/onnx/operators/onnx__Reshape.html
        return y.producer().attributes.get("allowzero", None) is None

    def rewrite(self, op, x, shape, y):
        # Insert Reshape with explicitly set allowzero attribute
        return op.Reshape(x, shape, allowzero=0)


# Eliminates reshapes without effect from the graph, i.e., those where the
# output shape is the same as the input shape
@passes.verify.equality
@passes.register("reorder")
class EliminateIdentityReshape(Transformation, RewriteRulePass):
    def pattern(self, op, x, shape):
        return op.Reshape(x, shape, _outputs=["y"])

    def check(self, op, x, shape, y):
        if x.shape is not None and y.shape is not None:
            if x.shape.is_static() and y.shape.is_static():
                return np.all(x.shape.numpy() == y.shape.numpy())
        return False

    def rewrite(self, op, x, shape, y):
        return op.Identity(x)


# Matching against one value pattern from a selection of alternative patterns,
# constructing named values and attributes to be matched
from onnxscript.rewriter._pattern_ir import (  # noqa: Protected module...
    OrValue, ValuePattern, AttrPattern
)

# Scalars allow for some simplification, e.g., trivial broadcasting
from onnx_passes.passes.util import is_scalar, is_constant

# Type hinting callables as transformation template arguments/placeholders
from typing import Callable

# Transformation templates are implemented by inspecting the signature of the
# operator-specializing function
import inspect


# # Transformation template matching elementwise n-ary operators with matching
# # reshapes at the inputs: The order of reshape and elementwise operators can be
# # switched whenever all inputs and thus the output of the elementwise operation
# # have the same shape, i.e., the operation does not broadcast any dimension.
# #
# # The template takes care of transferring attributes from the matched to the
# # replacement elementwise operator.
# #
# # Note: In case of unary elementwise operators the match condition is trivially
# # fulfilled and no extra Reshape is necessary.
# class _MoveReshapePastElementwise:
#     # Elementwise operator template to be filled in by the template
#     # specialization: Callable accepting self, the op, the inputs and **kwargs
#     __operator__: Callable
#
#     # Extracts parameters from the operator template which are supposed to be
#     # matched to the pattern
#     @property
#     def parameters(self):
#         # Inspect the function signature of the template specialization to
#         # derive the list of input names
#         parameters = inspect.signature(self.__operator__).parameters.values()
#         # Remove all keyword only arguments, these are reserved as auxiliary
#         # variables during the matching process, i.e., interpretation is up to
#         # the template specialization
#         parameters = [param.name for param in parameters if param.kind not in {
#             inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD
#         }]
#         # Drop first two (actually only the first as the method is bound) which
#         # are always (self, op)
#         return parameters[1:]
#
#     # Extracts parameters from the operator template which are not supposed to
#     # be matched to the pattern and only serve auxiliary purposes
#     @property
#     def auxiliaries(self):
#         # Inspect the function signature of the template specialization to
#         # derive the list of input names
#         parameters = inspect.signature(self.__operator__).parameters.values()
#         # Keep required keyword only arguments, these are reserved as auxiliary
#         # variables during the matching process, i.e., interpretation is up to
#         # the template specialization
#         return [param.name for param in parameters if param.kind in {
#             inspect.Parameter.KEYWORD_ONLY
#         }]


# Checks whether the two shapes are related via squeeze/unsqueeze and if so,
# extracts the squeezed/unsqueezed axes
def _squeezed_or_unsqueezed(shape, other):
    # Iteration state: Keeps track of current dimension index in each shape, the
    # list of squeezed/unsqueezed dimensions and whether this still describes a
    # valid squeeze/unsqueeze relation between shapes
    i, j, squeezed, unsqueezed, valid = 0, 0, [], [], True

    # Keep advancing until both shapes are fully processes - indices might
    # diverge over time, within bounds access is ensured
    while i < len(shape) or j < len(other):
        # Both indices still within bounds
        if i < len(shape) and j < len(other):
            # If shapes match at these indices, this is a dimension in common,
            # just advance the indices, this is neither squeezed nor unsqueezed
            if shape[i] == other[j]:
                i += 1
                j += 1
                # Continue with the next position
                continue

        # Shape still within bounds
        if i < len(shape):
            # Dimensions of size 1 are squeezed (missing from other shape)
            if shape[i] == 1:
                # Track squeezed dimensions
                squeezed.append(i)
                # Advance index of the processed dimension
                i += 1
                # Continue with the next position
                continue

        # Other shape still within bounds
        if j < len(other):
            # Track unsqueezed dimensions
            unsqueezed.append(j)
            # Advance index of the processed dimension
            j += 1
            # Continue with the next position
            continue

        # Dimensions did not match (or we are already out of bounds for one of
        # the shapes) - these shapes are not related via squeeze/unsqueeze
        valid = False
        # Break to not end up in endless loop. Also, once invalid will never be
        # valid again
        break

    # A tensor of shape can be transformed into other by successive squeeze and
    # unsqueeze if valid
    return squeezed, unsqueezed, valid


# Transformation template matching elementwise n-ary operators with matching
# reshapes at the output: The order of reshape and elementwise operators can be
# switched whenever all inputs and thus the output of the elementwise operation
# have the same shape, i.e., the operation does not broadcast any dimension.
#
# The template takes care of transferring attributes from the matched to the
# replacement elementwise operator.
#
# Note: In case of unary elementwise operators the match condition is trivially
# fulfilled and no extra Reshape is necessary.
class _MoveElementwisePastReshape(Transformation, RewriteRulePass):
    # Elementwise operator template to be filled in by the template
    # specialization: Callable accepting self, the op, the inputs and **kwargs
    __operator__: Callable

    # Extracts parameters from the operator template which are supposed to be
    # matched to the pattern
    @property
    def parameters(self):
        # Inspect the function signature of the template specialization to
        # derive the list of input names
        parameters = inspect.signature(self.__operator__).parameters.values()
        # Remove all keyword only arguments, these are reserved as auxiliary
        # variables during the matching process, i.e., interpretation is up to
        # the template specialization
        parameters = [param.name for param in parameters if param.kind not in {
            inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD
        }]
        # Drop first two (actually only the first as the method is bound) which
        # are always (self, op)
        return parameters[1:]

    # Extracts parameters from the operator template which are not supposed to
    # be matched to the pattern and only serve auxiliary purposes
    @property
    def auxiliaries(self):
        # Inspect the function signature of the template specialization to
        # derive the list of input names
        parameters = inspect.signature(self.__operator__).parameters.values()
        # Keep required keyword only arguments, these are reserved as auxiliary
        # variables during the matching process, i.e., interpretation is up to
        # the template specialization
        return [param.name for param in parameters if param.kind in {
            inspect.Parameter.KEYWORD_ONLY
        }]

    def pattern(self, op, shape):
        # For each parameter expected by the template specialization create a
        # matchable input value pattern - these are like dynamic arguments
        xs = [ValuePattern(x) for x in self.parameters]
        # Forward all inputs to the template specialization operator
        return op.Reshape(self.__operator__(op, *xs, _outputs=["_out"]), shape)

    def check(self, op, _out, shape, **kwargs):
        # The output shape produced by Reshape must be a constant to check for
        # shape-compatibility
        if (shape := ir.convenience.get_const_tensor(shape)) is None:
            return False

        # For each input check for the shape being compatible with the output of
        # the reshaped elementwise operator
        for i, x in enumerate(self.parameters):
            # The input shape must be a constant to check shape-compatibility
            if kwargs[x].shape is None or kwargs[x].shape.is_dynamic():
                return False

        # Count the number of constant inputs and reject the transformation if
        # this would result in more non-constant foldable reshapes
        return sum(not is_constant(kwargs[x]) for x in self.parameters) <= 1

    def rewrite(self, op, _out, shape, **kwargs):
        # Collect a list of replacement inputs (generate graph patterns of
        # Squeeze/Unsqueeze equivalents)
        xs = []

        # Calculate output shape of the elementwise operator following
        # broadcasting
        broadcast = np.broadcast_shapes(
            *[kwargs[x].shape for x in self.parameters]
        )

        # For each of the inputs a new Reshape operation will be inserted in
        # front of the elementwise operation
        for x in [kwargs[x] for x in self.parameters]:
            # Scalars are trivial to broadcast without reshaping, so this can be
            # skipped
            # Note: This is necessary to treat Clip as an elementwise operation,
            # which it is not exactly as min and max do not broadcast
            if not is_scalar(x):
                # Expand and reshape each input to the elementwise operation,
                # broadcasting is now fully explicit
                x = op.Reshape(
                    op.Expand(x, op.Constant(value_ints=broadcast)), shape
                )

            # Collect replacement input to be wired up with the elementwise
            # operator down below
            xs.append(x)

        # Collect auxiliary arguments used by the template specialization which
        # are matched by the pattern but not as part of the interface inputs
        aux = {key: kwargs[key] for key in self.auxiliaries if key in kwargs}
        # Forward the output capture if the template operator lists this among
        # the auxiliaries
        aux = {**aux, **({"_out": _out} if "_out" in self.auxiliaries else {})}
        # Combine auxiliaries withe attributes transferred from the original
        # operator of the matched pattern
        attributes = {**aux, **_out.producer().attributes}
        # Expand inputs, attributes and auxiliaries into the operator
        return self.__operator__(op, *xs, **attributes)


@passes.verify.equality  # noqa: Seems like duplicate but it is not...
@passes.register("reorder")
class MoveAddPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Add(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveSubPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Sub(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveMulPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Mul(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveSquarePastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Mul(x, x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveSiluPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Mul(op.Sigmoid(x), x, **kwargs)

    @property
    def commute(self):
        return True

@passes.verify.equality
@passes.register("reorder")
class MoveDivPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Div(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveBitwiseOrPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.BitwiseOr(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveBitwiseAndPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.BitwiseAnd(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveBitwiseXorPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.BitwiseXor(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveBitShiftPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.BitShift(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveOrPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Or(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveAndPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.And(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveXorPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Xor(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveEqualPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Equal(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveLessPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Less(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveLessOrEqualPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.LessOrEqual(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveGreaterPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Greater(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveGreaterOrEqualPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.GreaterOrEqual(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveModPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Mod(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MovePowPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Pow(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MovePReluPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.PRelu(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveAbsPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Abs(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveAcosPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Acos(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveAcoshPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Acosh(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveAsinPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Asin(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveAsinhPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Asinh(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveAtanPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Atan(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveAtanhPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Atanh(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveBitwiseNotPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.BitwiseNot(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveCastPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Cast(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveCeilPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Ceil(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveCeluPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Celu(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveCosPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Cos(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveCoshPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Cosh(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveEluPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Elu(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveErfPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Erf(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveExpPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Exp(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveFloorPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Floor(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveGeluPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Gelu(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveHardSigmoidPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.HardSigmoid(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveHardSwishPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.HardSwish(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveIdentityPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Identity(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveIfInfPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.IfInf(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveIsNaNPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.IsNaN(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveLeakyReluPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.LeakyRelu(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveLogPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Log(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveMishPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Mish(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveNegPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Neg(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveNotPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Not(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveReciprocalPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Reciprocal(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveReluPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Relu(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveRoundPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Round(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveSeluPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Selu(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveShrinkPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Shrink(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveSigmoidPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Sigmoid(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveSignPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Sign(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveSinPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Sin(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveSinhPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Sinh(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveSoftplusPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Softplus(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveSoftsignPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Softsign(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveSqrtPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Sqrt(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveTanPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Tan(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveTanhPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Tanh(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveThresholdedReluPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, **kwargs: \
        op.ThresholdedRelu(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveWherePastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, condition, x, y, **kwargs: \
        op.Where(condition, x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveClipPastReshape(_MoveElementwisePastReshape):
    __operator__ = lambda _, op, x, _min, _max, **kwargs: \
        op.Clip(x, _min, _max, **kwargs)


# Moves Transpose operations past Reshape operation if the reshape only squeezes
# or unsqueezes axes -  in these cases, corresponding permutation axes can be
# inserted or deleted to switch the oder of operations
@passes.verify.equality
@passes.register("reorder")
class MoveTransposePastReshape(Transformation, RewriteRulePass):
    def pattern(self, op, x, perm, shape):
        return op.Reshape(op.Transpose(x, perm=perm, _outputs=["_out"]), shape)

    def check(self, op, x, perm, shape, _out, **kwargs):
        # The output shape produced by Reshape must be a constant to check for
        # shape-compatibility
        if (shape := ir.convenience.get_const_tensor(shape)) is None:
            return False

        # The input shape must be a constant to check shape-compatibility
        if _out.shape is None:
            return False

        # Check whether the reshape is a composition of squeeze and unsqueeze
        # operations, i.e., only adds and deletes axes of size 1
        if not _squeezed_or_unsqueezed(_out.shape, shape.numpy())[-1]:
            return False

        # Shapes are compatible - accept the pattern for rewrite if the
        # permutation is available
        return perm is not None and perm.as_ints() is not None

    def rewrite(self, op, x, perm, shape, _out):
        # Constant shape produced by reshape as numpy array for calculating
        # squeeze/unsqueeze equivalents
        shape = ir.convenience.get_const_tensor(shape).numpy()

        # Decompose the reshape operation into the squeezing and unsqueezing
        # component
        squeeze, unsqueeze, _ = _squeezed_or_unsqueezed(_out.shape, shape)

        # Apply the permutation to the axes as the transpose will be placed
        # after the Reshape, but axes has been derived with Reshape before
        squeeze = [perm.as_ints()[i] for i in squeeze]

        # Squeeze must be applied first, do not insert Squeeze operator with
        # empty axes
        if squeeze:
            x = op.Squeeze(x, op.Constant(value_ints=squeeze))

        # Delete squeezed axes from permutation list without adjusting the index
        # there might be holes now
        perm = [i for i in perm.as_ints() if i not in squeeze]
        # Fill holes in permutation indices by remapping the indices to the new
        # shorter dimensions
        perm = [sorted(perm).index(i) for i in perm]

        # Adjust the indices of unsqueezed axes to inert holes to be filled
        # with new axes
        for u in unsqueeze:
            perm = [i if i < u else i + 1 for i in perm]

        # Insert new unsqueezed axes into the holes of the permutation list
        for u in unsqueeze:
            perm.insert(u, u)

        # Apply the permutation to the axes as the transpose will be placed
        # after the Reshape, but axes has been derived with Reshape before
        unsqueeze = [perm[i] for i in unsqueeze]

        # Unsqueeze must be applied second, do not insert Unsqueeze operator
        # with empty axes
        if unsqueeze:
            x = op.Unsqueeze(x, op.Constant(value_ints=unsqueeze))

        # Insert new permutation un squeezed/unsqueezed axes
        return op.Transpose(x, perm=perm)

# TODO: Implement reshape MatMul and reduction (if applicable) segments of the
#  graph
