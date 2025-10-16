# ir.Value
import onnx_ir as ir

# Matching against one value pattern from a selection of alternative patterns
from onnxscript.rewriter.pattern import OrValue

# Algebraic properties as transformation templates
from onnx_passes.passes.streamline.algebraic._properties import (
    _Associative,
    _DistributiveLhs,
    _DistributiveRhs,
)

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# All algebraic passes are transformations derived from pattern-based rewrite
# rules
from onnx_passes.passes.base import Transformation, RewriteRulePass, \
    RewriteRuleSetPass

# Checking ir.Value for being constants and comparing constants to be identical
from onnx_passes.passes.util import identical_constants, is_constant, is_scalar


# ==============================================================================
# Matrix multiplication is constrained to numeric input and output tensors of
# the same type. Assuming floating-point arithmetic approximately associative,
# the following properties are exploited to simplify expressions, and to group,
# propagate, fuse and eventually eliminate constants:
#
# Associativity, distributivity over addition, commutativity of scalar
# multiplication, matching transpose reversing the order.
#
# Matrix multiplication itself is non-commutative, thus most of the following
# transformations are not tagged to commute.
#
# As floating-point arithmetic is only approximately associative, all these
# transformations must be tagged @passes.verify.tolerance instead of equality.
# ==============================================================================

@passes.verify.tolerance
@passes.register("algebraic")
class GroupMatMul(_Associative):
    __OP__ = lambda _, op, x, y: op.MatMul(x, y)


@passes.verify.tolerance
@passes.register("algebraic")
class DistributiveMatMulAddLhs(_DistributiveLhs):
    __MUL__ = lambda _, op, x, y: op.MatMul(x, y)
    __ADD__ = lambda _, op, x, y: op.Add(x, y)


@passes.verify.tolerance
@passes.register("algebraic")
class DistributiveMatMulAddRhs(_DistributiveRhs):
    __MUL__ = lambda _, op, x, y: op.MatMul(x, y)
    __ADD__ = lambda _, op, x, y: op.Add(x, y)


# Commutativity of scalar multiplication for numeric tensors combined with
# associativity of matrix multiplication: (ax) @ y = a(x @ y)
@passes.verify.tolerance
@passes.register("algebraic")
class MoveMulPastMatMulLhs(Transformation, RewriteRulePass):
    # Commutativity here applies to the scalar multiplication, i.e., op.Mul, not
    # the matrix multiplication
    @property
    def commute(self):
        return True

    def pattern(self, op, x, y, a):
        return op.MatMul(op.Mul(a, x), y)

    def check(self, op, x, y, a):
        return is_scalar(a)

    def rewrite(self, op, x, y, a):
        return op.Mul(a, op.MatMul(x, y))


# Commutativity of scalar multiplication for numeric tensors combined with
# associativity of matrix multiplication: x @ (ay) = a(x @ y)
@passes.verify.tolerance
@passes.register("algebraic")
class MoveMulPastMatMulRhs(Transformation, RewriteRulePass):
    # Commutativity here applies to the scalar multiplication, i.e., op.Mul, not
    # the matrix multiplication
    @property
    def commute(self):
        return True

    def pattern(self, op, x, y, a):
        return op.MatMul(x, op.Mul(a, y))

    def check(self, op, x, y, a):
        return is_scalar(a)

    def rewrite(self, op, x, y, a):
        return op.Mul(a, op.MatMul(x, y))


# Moves elementwise addition past matrix multiplication: Distributivity... but
# reversed direction - this only makes sense with two constant inputs.
#
# This transforms (x + b) @ w => x @ w + b @ w where b and w hint at the usual
# interpretation of constant bias and weight tensors. Also handles the case
# where w is on the left (MatMul is non-commutative!) via a rewrite rule set.
#
# Note: Should be followed by un-broadcasting optimization to reduce potential
# parameter tensor size increase due to broadcasting required to have valid
# shapes after reordering.
@passes.verify.tolerance
@passes.register("algebraic")
class MoveAddPastMatMul(Transformation, RewriteRuleSetPass):
    # Commutativity here applies to the scalar addition, i.e., op.Add, not
    # the matrix multiplication
    @property
    def commute(self):
        return True

    def pattern(self):
        return [
            lambda op, x, w, b: op.MatMul(op.Add(x, b), w),
            lambda op, x, w, b: op.MatMul(w, op.Add(x, b)),
        ]

    def check(self):
        return [
            lambda op, x, w, b: is_constant(w) and is_constant(b),
            lambda op, x, w, b: is_constant(w) and is_constant(b)
        ]

    def rewrite(self):
        # Explicitly expand the constant addition input to valid broadcasting,
        # matching the matrix multiplication input
        return [
            lambda op, x, w, b: op.Add(
                op.MatMul(x, w),
                op.MatMul(op.Expand(b, op.Shape(op.Add(x, b))), w)
            ),
            lambda op, x, w, b: op.Add(
                op.MatMul(w, x),
                op.MatMul(w, op.Expand(b, op.Shape(op.Add(x, b))))
            )
        ]


# Commutativity of scalar multiplication: If the rank of the multiplication
# input is 1, i.e., acting along the last axis, and these scales as well as the
# other input to the matrix multiplication are constant, simplify by absorbing
# the scales into the matrix multiplication.
@passes.verify.tolerance
@passes.register("algebraic")
class AbsorbMulIntoMatMul(Transformation, RewriteRuleSetPass):
    # Commutativity here applies to the scalar multiplication, i.e., op.Mul, not
    # the matrix multiplication
    @property
    def commute(self):
        return True
    
    def pattern(self):
        return [
            lambda op, x, a, b: op.MatMul(op.Mul(a, x), b),
            lambda op, x, a, b: op.MatMul(b, op.Mul(a, x)),
        ]

    def check(self):
        # Scales must have static shape of rank 1 or less, scales and other
        # MatMul input must be constants
        def _check(op, x, a, b):
            if a.shape is not None and a.shape.is_static():
                return is_constant(a) and is_constant(b) and a.shape.rank() <= 1
            return False

        # Both sides have the same match condition, just reorder the inputs to
        # the pattern (input "a" always refers to the multiplication scales)
        return [_check, _check]

    def rewrite(self):
        # Reshape the scales to have valid broadcasting matching the matrix
        # multiplication input
        return [
            lambda op, x, a, b: op.MatMul(
                x, op.Mul(op.Reshape(a, op.Constant(value_ints=[-1, +1])), b)
            ),
            lambda op, x, a, b: op.MatMul(
                op.Mul(b, op.Reshape(a, op.Constant(value_ints=[+1, -1]))), x
            ),
        ]


# Commutativity of scalar multiplication allows to propagate transposes by
# reversing the order of multiplication - transposes must match in permutation
@passes.verify.tolerance
@passes.register("algebraic")
class MoveTransposePastMatMul(Transformation, RewriteRulePass):
    def pattern(self, op, x, y, perm_x, perm_y):
        return op.MatMul(
            OrValue([op.Transpose(x, perm=perm_x), op.Transpose(x)]),
            OrValue([op.Transpose(y, perm=perm_y), op.Transpose(y)]),
        )

    @staticmethod
    def _get_perms(x, y, perm_x, perm_y):
        # If no permutation is given, assumed reversing the dimensions, see
        #   https://onnx.ai/onnx/operators/onnx__Transpose.html
        if perm_x is None:
            perm_x = ir.Attr("perm_x", ir.AttributeType.INTS, [
                i for i in reversed(range(len(x.shape)))
            ])

        if perm_y is None:
            perm_y = ir.Attr("perm_y", ir.AttributeType.INTS, [
                i for i in reversed(range(len(y.shape)))
            ])

        return perm_x.as_ints(), perm_y.as_ints()

    def check(self, op, x, y, perm_x, perm_y):
        # Resolve the permutation attributes considering defaults for missing
        # attributes - these defaults depend on the input shapes
        perm_x, perm_y = self._get_perms(x, y, perm_x, perm_y)

        # The output tensor has as many dimensions as the longer input tensor
        # according to matrix multiplication broadcasting semantics
        output_dims = max(len(x.shape), len(y.shape))

        # If a permutation permutes fewer than the output dimensions, adjust
        # indices for matching by adding the number of missing dimensions
        if len(perm_x) < output_dims:
            perm_x = [i + output_dims - len(perm_x) for i in perm_x]

        if len(perm_y) < output_dims:
            perm_y = [i + output_dims - len(perm_y) for i in perm_y]

        # Broadcasting semantics right-align the shapes, thus match permutation
        # indices of all common dimensions from the right, i.e., reversed order
        return all(i == j for i, j in zip(reversed(perm_x), reversed(perm_y)))

    def rewrite(self, op, x, y, perm_x, perm_y):
        # Resolve the permutation attributes considering defaults for missing
        # attributes - these defaults depend on the input shapes
        perms = self._get_perms(x, y, perm_x, perm_y)
        # Insert Transpose past MatMul with combined matching permutation, which
        # is the permutation corresponding to the input with more dimensions
        return op.Transpose(
            op.MatMul(y, x), perm=perms[len(y.shape) >= len(x.shape)]
        )

# TODO: Consider adding a MoveMatMulPastTranspose matching on constant inputs to
#  the MatMul which could be constant-folded with the Transpose
