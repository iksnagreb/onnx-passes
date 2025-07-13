# Matching against one value pattern from a selection of alternative patterns,
# constructing named values and attributes to be matched
from onnxscript.rewriter._pattern_ir import (  # noqa: Protected module...
    OrValue, ValuePattern, AttrPattern
)

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# All algebraic passes are transformations derived from pattern-based rewrite
# rules
from onnx_passes.passes.base import Transformation, RewriteRulePass

# Checking ir.Value for being constants and comparing constants to be identical
from onnx_passes.passes.util import is_constant, is_scalar

# Transformation templates are implemented by inspecting the signature of the
# operator-specializing function
import inspect

# NumPy used for calculations on shapes and constant tensors in rewrites and
# match conditions
import numpy as np


# Turns default permutation attribute which implicitly reverse the dimensions
# into explicit permutation to reduce the number of rules needed below...
@passes.verify.equality
@passes.register("streamline-shapes")
class InferTransposePerm(Transformation, RewriteRulePass):
    def pattern(self, op, x):
        return op.Transpose(x, _outputs=["_out"])

    def check(self, op, x, _out):
        if (_ := _out.producer().attributes.get("perm", None)) is None:
            return x.shape is not None
        return False

    def rewrite(self, op, x, _out):
        return op.Transpose(x, perm=list(reversed(range(len(x.shape)))))


# Fuses two consecutive transpose operations by applying the second permutation
# to the first permutation, effectively eliminating a transpose from the graph.
@passes.verify.equality
@passes.register("streamline-shapes")
class FuseTranspose(Transformation, RewriteRulePass):
    def pattern(self, op, x, perm1, perm2):
        return op.Transpose(op.Transpose(x, perm=perm1), perm=perm2)

    def rewrite(self, op, x, perm1, perm2):
        return op.Transpose(
            x, perm=[perm1.as_ints()[i] for i in perm2.as_ints()]
        )


# Eliminates transposes without effect from the graph, i.e., those where the
# permutation lists the dimensions in order
@passes.verify.equality
@passes.register("streamline-shapes")
class EliminateIdentityTranspose(Transformation, RewriteRulePass):
    def pattern(self, op, x, perm):
        return op.Transpose(x, perm=perm)

    def check(self, op, x, perm):
        return np.all(perm.as_ints() == list(range(len(perm.as_ints()))))

    def rewrite(self, op, x, perm):
        return x


# ==============================================================================
# The following is the forward direction of streamlining transposes - moving
# them downstream towards the end of the graph, i.e., from the inputs to the
# output of elementwise operators. Both directions could be relevant depending
# on the model, though the downstream direction is probably the preferable.
# ==============================================================================


# Transformation template matching elementwise n-ary operators with matching
# transposes at the inputs: Transpose "respects" these types of operators and
# the order can be swapped. Constants can be implicitly transposed, i.e., there
# can be no Transpose node, resulting in a matching constant-foldable transpose
# inserted at the constant input. Valid broadcasting of the inputs is ensured
# which might result in Unsqueeze operators inserted at the inputs.
#
# The template takes care of transferring attributes from the matched to the
# replacement elementwise operator.
#
# Note: In case of unary elementwise operators the match condition is trivially
# fulfilled and no extra Transpose or Unsqueeze is necessary.
class _MoveTransposePastElementwise(Transformation, RewriteRulePass):
    # Elementwise operator template to be filled in by the template
    # specialization: Callable accepting self, the op, the inputs and **kwargs
    __operator__: callable

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

    def pattern(self, op):
        # For each parameter expected by the template specialization create a
        # matchable input value pattern - these are like dynamic arguments
        xs = [ValuePattern(x) for x in self.parameters]
        # For each transposed input create a permutation attribute name
        perms = [AttrPattern(f"perm{i}") for i in range(len(xs))]
        # Generate transpose nodes for each of the inputs
        transposes = [op.Transpose(x, perm=perm) for x, perm in zip(xs, perms)]
        # For each input generate two options: Transposing and not transposing
        xs = [OrValue([transposed, x]) for transposed, x in zip(transposes, xs)]
        # Forward all inputs to the template specialization operator
        return self.__operator__(op, *xs, _outputs=["_out"])

    def check(self, op, **kwargs):
        # Collect all permutation attributes which are present
        perms = []

        # For each input check for the presence of a permutation attribute which
        # means this is a proper, transposed input.
        for i, x in enumerate(self.parameters):
            # Missing transpose is only allowed for constant inputs with known
            # shapes (or scalars)
            if (perm := kwargs.setdefault(f"perm{i}", None)) is None:
                if not is_constant(kwargs[x]) or kwargs[x].shape is None:
                    if not is_scalar(kwargs[x]):
                        return False
                continue
            # Collect permutations for further checks
            perms.append(perm.as_ints())

        # There must be at least one Transposed input for this to apply...
        if len(perms) < 1:
            return False

        # Check each pairing of permutations: Common dimensions must match
        for perm1 in perms:
            for perm2 in perms:
                # Filter for common dimensions from each permutation
                perm1 = [i for i in perm1 if i < len(perm2)]
                perm2 = [i for i in perm2 if i < len(perm1)]
                # A single mismatch for a single pair is enough to reject
                if not np.all(perm1 == perm2):
                    return False

        # All transposes have matching permutations
        return True

    def rewrite(self, op, _out, **kwargs):
        # Track the permutation with the largest number of permuted dimensions,
        # start with none
        perm = []

        # Find the overall output permutation of the operator pattern from the
        # matched permutation attributes and constant shapes
        for i, x in enumerate(self.parameters):
            # Get the matched input (proper input or constant) corresponding to
            # this parameter and the permutation if a transpose is present
            x, permx = kwargs[x], kwargs.setdefault(f"perm{i}", None)

            # This is a proper input. Track the permutation of the largest rank
            if permx is not None:
                if len(permx.as_ints()) >= len(perm):
                    perm = permx.as_ints()
            # This is a constant input. However, it might be the one determining
            # the output rank - the permutation must be adjusted accordingly
            else:
                # Add number of dimensions accounting for the difference in
                # tensor rank: x is a constant, assume the shape to be present
                padding = range(len(x.shape) - len(perm))
                # Add the padding dimensions from the left and adjust existing
                # indices to reflect extra dimensions
                perm = [*padding, *[i + len(padding) for i in perm]]

        # Collect replacements for the inputs of the elementwise operator
        xs = []

        # Rank of the output tensor: according to multidirectional broadcasting
        # this will be the largest rank of the inputs - matches the permutation
        rank_o = len(perm)

        # Each parameter either corresponds to a transposed proper input or not
        # transposed constant input: To ensure valid broadcasting after removing
        # transposes, some dimensions need to be added and the constants need to
        # be transposed.
        for i, x in enumerate(self.parameters):
            # Get the matched input (proper input or constant) corresponding to
            # this parameter and the permutation if a transpose is present
            x, permx = kwargs[x], kwargs.setdefault(f"perm{i}", None)

            # If a permutation is present, this is a proper input and any
            # broadcastable dimensions must be unsqueezed from the right
            if permx is not None:
                # Rank of the currently treated input x can be derived from the
                # permutation attribute
                rank_x = len(permx.as_ints())

                # If the graph is in a valid state, inputs are broadcastable,
                # and there are two mutually exclusive cases now:
                # 1. rank(x) < rank(o): Not enough axes to permute,
                #   introduce matching broadcastable dimensions
                if 1 < rank_x < rank_o:
                    x = op.Unsqueeze(
                        x, op.Constant(value_ints=[
                            i for i in perm[:rank_o - rank_x]
                        ])
                    )
                # 2. rank(x) = rank(o): Same number of axes on both sides,
                #   same permutation applies to both sides
                # x = x...

            # If no permutation is present, this must be a constant input where
            # the rank must be adjusted according to broadcasting rules before
            # inserting a transpose, i.e., squeezing/unsqueezing from the left
            if permx is None:
                # Rank of the currently treated input x can be derived from the
                # constant input shape
                rank_x = len(x.shape)

                # If the graph is in a valid state, inputs are broadcastable,
                # and there are two mutually exclusive cases now:
                # 1. rank(x) < rank(o): Not enough axes to permute, introduce
                #   matching broadcastable dimensions
                if 1 <= rank_x < rank_o:
                    x = op.Unsqueeze(
                        x, op.Constant(value_ints=range(rank_o - rank_x))
                    )
                # 2. rank(x) = rank(o): Same number of axes on both sides,
                #   same permutation applies to both sides
                # x = x...

                # Insert a constant-foldable transpose between the constant and
                # the corresponding input to the elementwise operator
                if 1 <= rank_x:
                    x = op.Transpose(x, perm=perm)
            # Collect all inputs to wire them up later
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
        # Expand inputs, attributes and auxiliaries into the operator followed
        # by transpose
        return op.Transpose(self.__operator__(op, *xs, **attributes), perm=perm)


# ==============================================================================
# The following instantiates the _MoveTransposePastElementwise transformation
# template explicitly for each operator. In the future this might be condensed
# to a single specialization per category (unary, binary, ternary) to reduce the
# amount of code - especially as this does not scale when refactoring, e.g., to
# add shared specialization per category...
#
# However, for now this looks rather clean and results in one specialization for
# each concrete operator pattern which is also clearly reflected in the name of
# the transformation - when prototyping and debugging this allows picking and
# ordering transformations with fine granularity.
# ==============================================================================

@passes.verify.equality  # noqa: Seems like duplicate but it is not...
@passes.register("streamline-shapes")
class MoveTransposePastAdd(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Add(x, y, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastSub(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Sub(x, y, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastMul(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Mul(x, y, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastDiv(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Div(x, y, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastBitwiseOr(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.BitwiseOr(x, y, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastBitwiseAnd(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.BitwiseAnd(x, y, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastBitwiseXor(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.BitwiseXor(x, y, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastBitShift(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.BitShift(x, y, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastOr(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Or(x, y, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastAnd(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.And(x, y, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastXor(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Xor(x, y, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastEqual(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Equal(x, y, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastLess(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Less(x, y, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastLessOrEqual(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.LessOrEqual(x, y, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastGreater(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Greater(x, y, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastGreaterOrEqual(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.GreaterOrEqual(x, y, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastMod(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Mod(x, y, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastPow(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Pow(x, y, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastPRelu(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.PRelu(x, y, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastAbs(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Abs(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastAcos(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Acos(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastAcosh(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Acosh(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastAsin(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Asin(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastAsinh(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Asinh(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastAtan(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Atan(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastAtanh(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Atanh(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastBitwiseNot(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.BitwiseNot(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastCast(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Cast(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastCeil(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Ceil(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastCelu(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Celu(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastCos(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Cos(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastCosh(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Cosh(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastElu(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Elu(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastErf(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Erf(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastExp(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Exp(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastFloor(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Floor(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastGelu(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Gelu(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastHardSigmoid(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.HardSigmoid(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastHardSwish(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.HardSwish(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastIdentity(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Identity(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastIfInf(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.IfInf(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastIsNaN(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.IsNaN(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastLeakyRelu(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.LeakyRelu(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastLog(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Log(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastMish(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Mish(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastNeg(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Neg(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastNot(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Not(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastReciprocal(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Reciprocal(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastRelu(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Relu(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastRound(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Round(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastSelu(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Selu(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastShrink(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Shrink(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastSigmoid(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Sigmoid(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastSign(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Sign(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastSin(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Sin(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastSinh(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Sinh(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastSoftplus(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Softplus(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastSoftsign(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Softsign(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastSqrt(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Sqrt(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastTan(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Tan(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastTanh(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Tanh(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastThresholdedRelu(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, **kwargs: \
        op.ThresholdedRelu(x, **kwargs)


@passes.verify.equality
@passes.register("streamline-shapes")
class MoveTransposePastWhere(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, condition, x, y, **kwargs: \
        op.Where(condition, x, y, **kwargs)


@passes.verify.equality  # noqa: Seems like duplicate but it is not...
@passes.register("streamline-shapes")
class MoveTransposePastClip(_MoveTransposePastElementwise):
    __operator__ = lambda _, op, x, _min, _max, **kwargs: \
        op.Clip(x, _min, _max, **kwargs)


# TODO: elementwise n-ary operators with variadic inputs: Max, Mean, Min, Sum


# ==============================================================================
# The following experiments with a second layer of templates specializing the
# _MoveTransposePastElementwise transformation template for operator categories.
#
# This probably results in short and easier to maintain code, but abusing the
# submodule op with prefix pattern to match arbitrary operators and capturing
# auxiliary output arguments to detect the original operator feels ugly/hacky.
# ==============================================================================

# Categorization of elementwise operator types as unary, binary, ternary, etc.
from onnx_passes.traits.elementwise import (
    is_elementwise_unary, is_elementwise_binary, is_elementwise_ternary
)


# Instantiates the _MoveTransposePastElementwise to match any elementwise unary
# operator from a list of supported operators
@passes.verify.equality  # noqa: Seems like duplicate but it is not...
@passes.register()
class MoveTransposePastElementwiseUnary(_MoveTransposePastElementwise):
    def check(self, op, _out, **kwargs):  # noqa: Signature does not match super
        if _out is not None and is_elementwise_unary(_out.producer()):
            return super().check(op, **kwargs)
        return False

    @staticmethod
    def __operator__(op, x, *, _out=None, **kwargs):
        # Missing auxiliary argument selects pattern matching branch: Abuse the
        # submodule op with prefix pattern to match any unary operator
        if _out is None:
            return op.submodule("")(x, **kwargs)
        # Present auxiliary arguments selects the pattern rewrite branch: Use
        # the auxiliary variable to generate the same operator as matched before
        return op.op(_out.producer().op_type, [x], attributes=kwargs)


# Instantiates the _MoveTransposePastElementwise to match any elementwise binary
# operator from a list of supported operators
@passes.verify.equality
@passes.register()
class MoveTransposePastElementwiseBinary(_MoveTransposePastElementwise):
    def check(self, op, _out, **kwargs):  # noqa: Signature does not match super
        if _out is not None and is_elementwise_binary(_out.producer()):
            return super().check(op, **kwargs)
        return False

    @staticmethod
    def __operator__(op, x, y, *, _out=None, **kwargs):
        # Missing auxiliary argument selects pattern matching branch: Abuse the
        # submodule op with prefix pattern to match any unary operator
        if _out is None:
            return op.submodule("")(x, y, **kwargs)
        # Present auxiliary arguments selects the pattern rewrite branch: Use
        # the auxiliary variable to generate the same operator as matched before
        return op.op(_out.producer().op_type, [x, y], attributes=kwargs)


# Instantiates the _MoveTransposePastElementwise to match any elementwise
# ternary operator from a list of supported operators
@passes.verify.equality
@passes.register()
class MoveTransposePastElementwiseTernary(_MoveTransposePastElementwise):
    def check(self, op, _out, **kwargs):  # noqa: Signature does not match super
        if _out is not None and is_elementwise_ternary(_out.producer()):
            return super().check(op, **kwargs)
        return False

    @staticmethod
    def __operator__(op, x, y, z, *, _out=None, **kwargs):
        # Missing auxiliary argument selects pattern matching branch: Abuse the
        # submodule op with prefix pattern to match any unary operator
        if _out is None:
            return op.submodule("")(x, y, z, **kwargs)
        # Present auxiliary arguments selects the pattern rewrite branch: Use
        # the auxiliary variable to generate the same operator as matched before
        return op.op(_out.producer().op_type, [x, y, z], attributes=kwargs)


# ==============================================================================
# The following is the reverse direction of streamlining transposes - moving
# them upstream towards the start of the graph, i.e., from the output to the
# inputs of elementwise operators. Both directions could be relevant depending
# on the model, though the downstream direction is probably the preferable.
# ==============================================================================

# Transformation template matching elementwise n-ary operators followed by
# transpose at the output: Transpose "respects" these types of operators and
# the order can be swapped. Valid broadcasting of the inputs is ensured which
# might result in Unsqueeze operators inserted at the inputs.
#
# The template takes care of transferring attributes from the matched to the
# replacement elementwise operator.
#
# Note: In case of unary elementwise operators the match condition is trivially
# fulfilled and no extra Transpose or Unsqueeze is necessary.
class _MoveElementwisePastTranspose(Transformation, RewriteRulePass):
    # Elementwise operator template to be filled in by the template
    # specialization: Callable accepting self, the op, the inputs and **kwargs
    __operator__: callable

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

    def pattern(self, op, perm):
        # For each parameter expected by the template specialization create a
        # matchable input value pattern - these are like dynamic arguments
        xs = [ValuePattern(x) for x in self.parameters]
        # Forward all inputs to the template specialization operator
        return op.Transpose(
            self.__operator__(op, *xs, _outputs=["_out"]), perm=perm
        )

    def rewrite(self, op, perm, _out, **kwargs):
        # Collect replacements for the inputs of the elementwise operator
        xs = []

        # Rank of the output tensor: according to multidirectional broadcasting
        # this will be the largest rank of the inputs - matches the permutation
        rank_o = op.Constant(value_int=len(perm.as_ints()))

        # For each of the inputs a transpose and potential unsqueezing of axes
        # to ensure valid broadcasting must be inserted
        for x in [kwargs[x] for x in self.parameters]:
            # If the graph is in a valid state, inputs are broadcastable,
            # and there are two mutually exclusive cases now:
            x = op.Where(
                # 1. rank(x) < rank(o): Not enough axes to permute, introduce
                #   matching broadcastable dimensions
                op.Less(op.Size(op.Shape(x)), rank_o),
                op.Unsqueeze(
                    x, op.Range(
                        op.Constant(value_int=0),
                        op.Sub(rank_o, op.Size(op.Shape(x))),
                        op.Constant(value_int=1)
                    ),
                ),
                # 2. rank(x) = rank(o): Same number of axes on both sides,
                #   same permutation applies to both sides
                # x = x...
                x
            )

            # Transpose each individual input and collect for expansion into the
            # operator template
            xs.append(op.Transpose(x, perm=perm))

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


# ==============================================================================
# The following instantiates the _MoveElementwisePastTranspose transformation
# template explicitly for each operator. In the future this might be condensed
# to a single specialization per category (unary, binary, ternary) to reduce the
# amount of code - especially as this does not scale when refactoring, e.g., to
# add shared specialization per category...
#
# However, for now this looks rather clean and results in one specialization for
# each concrete operator pattern which is also clearly reflected in the name of
# the transformation - when prototyping and debugging this allows picking and
# ordering transformations with fine granularity.
# ==============================================================================


@passes.verify.equality  # noqa: Seems like duplicate but it is not...
@passes.register()
class MoveAddPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Add(x, y, **kwargs)


@passes.verify.equality
@passes.register()
class MoveSubPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Sub(x, y, **kwargs)


@passes.verify.equality
@passes.register()
class MoveMulPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Mul(x, y, **kwargs)


@passes.verify.equality
@passes.register()
class MoveDivPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Div(x, y, **kwargs)


@passes.verify.equality
@passes.register()
class MoveBitwiseOrPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.BitwiseOr(x, y, **kwargs)


@passes.verify.equality
@passes.register()
class MoveBitwiseAndPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.BitwiseAnd(x, y, **kwargs)


@passes.verify.equality
@passes.register()
class MoveBitwiseXorPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.BitwiseXor(x, y, **kwargs)


@passes.verify.equality
@passes.register()
class MoveBitShiftPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.BitShift(x, y, **kwargs)


@passes.verify.equality
@passes.register()
class MoveOrPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Or(x, y, **kwargs)


@passes.verify.equality
@passes.register()
class MoveAndPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.And(x, y, **kwargs)


@passes.verify.equality
@passes.register()
class MoveXorPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Xor(x, y, **kwargs)


@passes.verify.equality
@passes.register()
class MoveEqualPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Equal(x, y, **kwargs)


@passes.verify.equality
@passes.register()
class MoveLessPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Less(x, y, **kwargs)


@passes.verify.equality
@passes.register()
class MoveLessOrEqualPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.LessOrEqual(x, y, **kwargs)


@passes.verify.equality
@passes.register()
class MoveGreaterPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Greater(x, y, **kwargs)


@passes.verify.equality
@passes.register()
class MoveGreaterOrEqualPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.GreaterOrEqual(x, y, **kwargs)


@passes.verify.equality
@passes.register()
class MoveModPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Mod(x, y, **kwargs)


@passes.verify.equality
@passes.register()
class MovePowPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.Pow(x, y, **kwargs)


@passes.verify.equality
@passes.register()
class MovePReluPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, y, **kwargs: \
        op.PRelu(x, y, **kwargs)


@passes.verify.equality
@passes.register()
class MoveAbsPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Abs(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveAcosPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Acos(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveAcoshPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Acosh(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveAsinPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Asin(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveAsinhPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Asinh(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveAtanPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Atan(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveAtanhPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Atanh(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveBitwiseNotPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.BitwiseNot(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveCastPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Cast(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveCeilPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Ceil(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveCeluPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Celu(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveCosPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Cos(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveCoshPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Cosh(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveEluPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Elu(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveErfPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Erf(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveExpPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Exp(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveFloorPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Floor(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveGeluPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Gelu(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveHardSigmoidPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.HardSigmoid(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveHardSwishPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.HardSwish(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveIdentityPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Identity(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveIfInfPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.IfInf(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveIsNaNPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.IsNaN(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveLeakyReluPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.LeakyRelu(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveLogPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Log(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveMishPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Mish(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveNegPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Neg(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveNotPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Not(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveReciprocalPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Reciprocal(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveReluPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Relu(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveRoundPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Round(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveSeluPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Selu(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveShrinkPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Shrink(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveSigmoidPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Sigmoid(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveSignPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Sign(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveSinPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Sin(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveSinhPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Sinh(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveSoftplusPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Softplus(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveSoftsignPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Softsign(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveSqrtPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Sqrt(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveTanPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Tan(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveTanhPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.Tanh(x, **kwargs)


@passes.verify.equality
@passes.register()
class MoveThresholdedReluPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, **kwargs: \
        op.ThresholdedRelu(x, **kwargs)


# # TODO: Cannot handle Where as the template itself inserts Where...
# @passes.verify.equality
# @passes.register()
# class MoveWherePastTranspose(_MoveElementwisePastTranspose):
#     __operator__ = lambda _, op, condition, x, y, **kwargs: \
#         op.Where(condition, x, y, **kwargs)


@passes.verify.equality  # noqa: Seems like duplicate but it is not...
@passes.register()
class MoveClipPastTranspose(_MoveElementwisePastTranspose):
    __operator__ = lambda _, op, x, _min, _max, **kwargs: \
        op.Clip(x, _min, _max, **kwargs)


# ==============================================================================
# The following experiments with a second layer of templates specializing the
# _MoveElementwisePastTranspose transformation template for operator categories.
#
# This probably results in short and easier to maintain code, but abusing the
# submodule op with prefix pattern to match arbitrary operators and capturing
# auxiliary output arguments to detect the original operator feels ugly/hacky.
# ==============================================================================


# Instantiates the _MoveTransposePastElementwise to match any elementwise unary
# operator from a list of supported operators
@passes.verify.equality  # noqa: Seems like duplicate but it is not...
@passes.register()
class MoveElementwiseUnaryPastTranspose(_MoveElementwisePastTranspose):
    def check(self, op, _out, **kwargs):  # noqa: Signature does not match super
        if _out is not None and is_elementwise_unary(_out.producer()):
            return super().check(op, **kwargs)
        return False

    @staticmethod
    def __operator__(op, x, *, _out=None, **kwargs):
        # Missing auxiliary argument selects pattern matching branch: Abuse the
        # submodule op with prefix pattern to match any unary operator
        if _out is None:
            return op.submodule("")(x, **kwargs)
        # Present auxiliary arguments selects the pattern rewrite branch: Use
        # the auxiliary variable to generate the same operator as matched before
        return op.op(_out.producer().op_type, [x], attributes=kwargs)


# Instantiates the _MoveTransposePastElementwise to match any elementwise binary
# operator from a list of supported operators
@passes.verify.equality
@passes.register()
class MoveElementwiseBinaryPastTranspose(_MoveElementwisePastTranspose):
    def check(self, op, _out, **kwargs):  # noqa: Signature does not match super
        if _out is not None and is_elementwise_binary(_out.producer()):
            return super().check(op, **kwargs)
        return False

    @staticmethod
    def __operator__(op, x, y, *, _out=None, **kwargs):
        # Missing auxiliary argument selects pattern matching branch: Abuse the
        # submodule op with prefix pattern to match any unary operator
        if _out is None:
            return op.submodule("")(x, y, **kwargs)
        # Present auxiliary arguments selects the pattern rewrite branch: Use
        # the auxiliary variable to generate the same operator as matched before
        return op.op(_out.producer().op_type, [x, y], attributes=kwargs)


# Instantiates the _MoveTransposePastElementwise to match any elementwise
# ternary operator from a list of supported operators
@passes.verify.equality
@passes.register()
class MoveElementwiseTernaryPastTranspose(_MoveElementwisePastTranspose):
    def check(self, op, _out, **kwargs):  # noqa: Signature does not match super
        if _out is not None and is_elementwise_ternary(_out.producer()):
            # TODO: Cannot handle Where as the template itself inserts Where...
            if _out.producer().op_type != "Where":
                return super().check(op, **kwargs)
        return False

    @staticmethod
    def __operator__(op, x, y, z, *, _out=None, **kwargs):
        # Missing auxiliary argument selects pattern matching branch: Abuse the
        # submodule op with prefix pattern to match any unary operator
        if _out is None:
            return op.submodule("")(x, y, z, **kwargs)
        # Present auxiliary arguments selects the pattern rewrite branch: Use
        # the auxiliary variable to generate the same operator as matched before
        return op.op(_out.producer().op_type, [x, y, z], attributes=kwargs)
