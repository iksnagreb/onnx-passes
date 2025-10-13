# Matching against one value pattern from a selection of alternative patterns,
# constructing named values and attributes to be matched
from onnxscript.rewriter._pattern_ir import ValuePattern  # noqa: Protected...

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# All algebraic passes are transformations derived from pattern-based rewrite
# rules
from onnx_passes.passes.base import Transformation, RewriteRulePass

# Checking ir.Value for being constants and comparing constants to be identical
from onnx_passes.passes.util import is_constant

# Domain used by custom operators implemented with this library
from onnx_passes.ops import DOMAIN as CUSTOM_DOMAIN

# Transformation templates are implemented by inspecting the signature of the
# operator-specializing function
import inspect

# Type hinting callables as transformation template arguments/placeholders
from typing import Callable


# Moves elementwise operators past convolution input generation (Im2Col) by
# inserting Im2Col operators for each elementwise input and broadcasting all
# inputs to account for potential shape mismatch.
#
# Note: This only works in general as Im2Col does not handle the padding, which,
# in our case as defined in onnx_passes.ops.im2col, is always the case.
#
# Note: This is currently restricted to a single non-constant input to avoid
# instantiating multiple input generators.
#
# Note: Should be followed by un-broadcasting optimization to reduce potential
# parameter tensor size increase due to broadcasting required to have valid
# shapes after reordering.
class _MoveElementwisePastIm2Col(Transformation, RewriteRulePass):
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

    def pattern(self, op, indices, _out, kernel_shape, strides, dilations):
        # For each parameter expected by the template specialization create a
        # matchable input value pattern - these are like dynamic arguments
        xs = [ValuePattern(x) for x in self.parameters]
        # Forward all inputs to the template specialization operator and connect
        # explicit inputs to the Im2Col operator from our domain
        return op.Im2Col(
            # Template pattern with arbitrary, specialization defined inputs
            self.__operator__(op, *xs, _outputs=["_out"]),
            # Precomputed input generator access pattern
            indices,
            # Input generator attributes: Together with the input shape, these
            # can be used to derive the access pattern
            kernel_shape=kernel_shape,
            strides=strides,
            dilations=dilations,
            # Im2Col is specified in our own custom operator domain
            _domain=CUSTOM_DOMAIN
        )

    def check(self, op, **kwargs):
        # TODO: For now disable reordering more than one non-constant input to
        #  the overall pattern to avoid non-constant foldable Im2Col operators
        return sum(not is_constant(kwargs[x]) for x in self.parameters) <= 1

    def rewrite(
            self, op, indices, _out, kernel_shape, strides, dilations, **kwargs
    ):
        # Collect auxiliary arguments used by the template specialization which
        # are matched by the pattern but not as part of the interface inputs
        aux = {key: kwargs[key] for key in self.auxiliaries if key in kwargs}
        # Forward the output capture if the template operator lists this among
        # the auxiliaries
        aux = {**aux, **({"_out": _out} if "_out" in self.auxiliaries else {})}
        # Combine auxiliaries withe attributes transferred from the original
        # operator of the matched pattern
        attributes = {**aux, **_out.producer().attributes}

        # List for collecting original and replacement inputs
        tmp, xs = [], []

        # Collect all inputs into temporary used to express the broadcasting of
        # the inputs via shape calculations and op.Expand within the graph
        for x in [kwargs[x] for x in self.parameters]:
            tmp.append(x)

        # Generate replacement inputs with an Im2Col for each and a (hopefully
        # constant-foldable) auxiliary pattern to express the broadcasting
        for x in [kwargs[x] for x in self.parameters]:
            xs.append(
                op.Im2Col(
                    # Auxiliary pattern handling the broadcasting which is
                    # otherwise skipped by reordering
                    op.Expand(
                        x, op.Shape(self.__operator__(op, *tmp, **attributes))
                    ),
                    # Precomputed access pattern of the input generator is not
                    # modified, any shape mismatch is accounted for by Expand
                    indices,
                    # Attributes configuring the input generator are not
                    # modified, same reason as above...
                    kernel_shape=kernel_shape,
                    strides=strides,
                    dilations=dilations,
                    # Still this is defined in our custom operator domain
                    _domain=CUSTOM_DOMAIN
                )
            )

        # Unpack the replacement inputs into the elementwise operator and
        # re-insert the original operator attributes
        return self.__operator__(op, *xs, **attributes)


@passes.verify.equality  # noqa: Seems like duplicate but it is not...
@passes.register("reorder")
class MoveAddPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Add(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveSubPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Sub(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveMulPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Mul(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveDivPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Div(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveBitwiseOrPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.BitwiseOr(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveBitwiseAndPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.BitwiseAnd(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveBitwiseXorPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.BitwiseXor(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveBitShiftPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.BitShift(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveOrPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Or(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveAndPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.And(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveXorPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Xor(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveEqualPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Equal(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveLessPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Less(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveLessOrEqualPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.LessOrEqual(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveGreaterPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Greater(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveGreaterOrEqualPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.GreaterOrEqual(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveModPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Mod(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MovePowPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Pow(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MovePReluPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.PRelu(x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveAbsPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Abs(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveAcosPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Acos(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveAcoshPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Acosh(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveAsinPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Asin(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveAsinhPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Asinh(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveAtanPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Atan(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveAtanhPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Atanh(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveBitwiseNotPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.BitwiseNot(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveCastPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Cast(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveCeilPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Ceil(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveCeluPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Celu(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveCosPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Cos(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveCoshPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Cosh(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveEluPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Elu(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveErfPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Erf(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveExpPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Exp(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveFloorPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Floor(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveGeluPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Gelu(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveHardSigmoidPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.HardSigmoid(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveHardSwishPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.HardSwish(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveIdentityPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Identity(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveIfInfPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.IfInf(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveIsNaNPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.IsNaN(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveLeakyReluPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.LeakyRelu(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveLogPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Log(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveMishPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Mish(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveNegPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Neg(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveNotPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Not(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveReciprocalPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Reciprocal(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveReluPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Relu(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveRoundPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Round(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveSeluPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Selu(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveShrinkPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Shrink(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveSigmoidPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Sigmoid(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveSignPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Sign(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveSinPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Sin(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveSinhPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Sinh(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveSoftplusPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Softplus(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveSoftsignPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Softsign(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveSqrtPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Sqrt(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveTanPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Tan(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveTanhPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Tanh(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveThresholdedReluPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, **kwargs: \
        op.ThresholdedRelu(x, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveWherePastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, condition, x, y, **kwargs: \
        op.Where(condition, x, y, **kwargs)


@passes.verify.equality
@passes.register("reorder")
class MoveClipPastIm2Col(_MoveElementwisePastIm2Col):
    __operator__ = lambda _, op, x, _min, _max, **kwargs: \
        op.Clip(x, _min, _max, **kwargs)
