# ir.Value
import onnx_ir as ir

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# All streamlining passes are transformations derived from pattern-based rewrite
# rules
from onnx_passes.passes.base import Transformation, RewriteRulePass

# NumPy used for calculations on shapes and constant tensors in rewrites and
# match conditions
import numpy as np


# Eliminates slice without effect from the graph, i.e., those where the output
# shape is the same as the input shape and no dimension is reversed
@passes.verify.equality
@passes.register("reorder")
class EliminateIdentitySlice(Transformation, RewriteRulePass):
    def pattern(self, op, x, starts, ends, axes, steps):
        return op.Slice(x, starts, ends, axes, steps, _outputs=["y"])

    def check(self, op, x, starts, ends, axes, steps, y):
        # Static and identical input and output shape implies static slice
        # parameters
        if x.shape is not None and y.shape is not None:
            if x.shape.is_static() and y.shape.is_static():
                if np.all(x.shape.numpy() == y.shape.numpy()):
                    # Slicing backwards might keep the shape but rearrange the
                    # content, this should be rejected
                    if (steps := ir.convenience.get_const_tensor(
                            steps)) is not None:
                        return np.all(steps.numpy() == 1)
        return False

    def rewrite(self, op, x, starts, ends, axes, steps, y):
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


# Transformation template matching elementwise n-ary operators followed by Slice
# at the output: The order of slice and elementwise operators can be switched
# by slicing the inputs to the n-ary operator.
#
# The template takes care of transferring attributes from the matched to the
# replacement elementwise operator.
#
# Note: In case of unary elementwise operators the match condition is trivially
# fulfilled and no extra Reshape is necessary.
class _MoveElementwisePastSlice(Transformation, RewriteRulePass):
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

    def pattern(self, op, starts, ends, axes, steps):
        # For each parameter expected by the template specialization create a
        # matchable input value pattern - these are like dynamic arguments
        xs = [ValuePattern(x) for x in self.parameters]
        # Forward all inputs to the template specialization operator
        return op.Slice(self.__operator__(op, *xs, _outputs=["_out"]), starts,
                        ends, axes, steps)

    def check(self, op, _out, **kwargs):
        # The shapes of all inputs must be statically known to decide whether to
        # insert a Slice at the input as scalars cannot be sliced
        for i, x in enumerate(self.parameters):
            if kwargs[x].shape is None or kwargs[x].shape.is_dynamic():
                return False

        # Count the number of constant inputs and reject the transformation if
        # this would result in more non-constant foldable slices
        # TODO: Consider dropping this restriction as moving Slice upwards
        #  should always reduce the data volume even if not eliminated as const
        return sum(not is_constant(kwargs[x]) for x in self.parameters) <= 1

    def rewrite(self, op, _out, starts, ends, axes, steps, **kwargs):
        # Rank, i.e., number of dimensions, of the output tensor used to decide
        # which axes to keep
        rank_o = op.Size(op.Shape(_out))

        # Normalize the axes input: Turn all negative counting axes to positive
        # indices to simplify cases below
        axes = op.Where(
            op.Less(axes, op.Constant(value_int=0)), op.Add(axes, rank_o), axes
        )

        # Collect a list of replacement inputs (generate graph patterns of slice
        # operators on part of the axes for each input)
        xs = []

        # For each of the inputs a new Slice operation will be inserted in
        # front of the elementwise operation
        for x in [kwargs[x] for x in self.parameters]:
            # If the graph is in a valid state, inputs are broadcastable,
            # and there are two mutually exclusive cases now:
            # 1. rank(x) < rank(o): Not enough axes to slice, remove matching
            #   broadcastable dimensions from all parameter inputs to the slice
            # Rank, i.e., number of dimensions, of the input tensor used to
            # decide which axes to keep
            rank_x = op.Size(op.Shape(x))

            # 1.1 Subtract leading rank(o) - rank(x) dimensions from the axes
            # but keep negative indices for selecting from the other tensors
            _axes = op.Sub(axes, op.Sub(rank_o, rank_x))

            # 1.2 Select the axes indices to keep, i.e., the common axes of the
            # input and the output after broadcasting
            # Note: there seems to be no "if i not in axes" ONNX equivalent
            keep = op.NonZero(op.Not(op.Less(_axes, op.Constant(value_int=0))))
            keep = op.Reshape(keep, op.Constant(value_ints=[-1]))

            # 1.3 Select common axes from the parameter tensors starts, ends,
            # steps and axes (remove negative indices)
            _starts = op.GatherElements(starts, keep)
            _ends = op.GatherElements(ends, keep)
            _steps = op.GatherElements(steps, keep)
            _axes = op.GatherElements(_axes, keep)

            # Input replacement: Slice only selected axes adjusted for removed
            # dimension according to broadcasting
            if not is_scalar(x):
                x = op.Slice(x, _starts, _ends, _axes, _steps)

            # 2. rank(x) = rank(o): Same number of axes on both sides, same
            #   slice applies to both sides (handled implicitly above)
            # x = x...

            # Collect all inputs (replacement patterns) to wire them up later
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
class MoveAddPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Add(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveSubPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Sub(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveMulPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Mul(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveDivPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Div(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveBitwiseOrPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.BitwiseOr(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveBitwiseAndPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.BitwiseAnd(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveBitwiseXorPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.BitwiseXor(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveBitShiftPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.BitShift(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveOrPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Or(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveAndPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.And(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveXorPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Xor(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveEqualPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Equal(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveLessPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Less(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveLessOrEqualPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.LessOrEqual(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveGreaterPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Greater(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveGreaterOrEqualPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.GreaterOrEqual(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveModPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Mod(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MovePowPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Pow(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MovePReluPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.PRelu(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveAbsPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Abs(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveAcosPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Acos(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveAcoshPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Acosh(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveAsinPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Asin(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveAsinhPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Asinh(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveAtanPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Atan(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveAtanhPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Atanh(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveBitwiseNotPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.BitwiseNot(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveCastPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Cast(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveCeilPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Ceil(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveCeluPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Celu(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveCosPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Cos(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveCoshPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Cosh(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveEluPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Elu(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveErfPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Erf(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveExpPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Exp(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveFloorPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Floor(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveGeluPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Gelu(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveHardSigmoidPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.HardSigmoid(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveHardSwishPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.HardSwish(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveIdentityPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Identity(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveIfInfPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.IfInf(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveIsNaNPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.IsNaN(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveLeakyReluPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.LeakyRelu(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveLogPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Log(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveMishPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Mish(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveNegPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Neg(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveNotPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Not(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveReciprocalPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Reciprocal(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveReluPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Relu(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveRoundPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Round(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveSeluPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Selu(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveShrinkPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Shrink(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveSigmoidPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Sigmoid(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveSignPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Sign(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveSinPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Sin(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveSinhPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Sinh(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveSoftplusPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Softplus(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveSoftsignPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Softsign(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveSqrtPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Sqrt(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveTanPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Tan(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveTanhPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Tanh(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveThresholdedReluPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, **kwargs: \
        op.ThresholdedRelu(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveWherePastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, condition, x, y, **kwargs: \
        op.Where(condition, x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveClipPastSlice(_MoveElementwisePastSlice):
    __operator__ = lambda _, op, x, _min, _max, **kwargs: \
        op.Clip(x, _min, _max, **kwargs)

# TODO: Slice-Reshape? Slice following MatMul? Slice-Split? Transpose-Slice?
