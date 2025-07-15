# ir.Model, ir.passes.PassResult, ir.from_proto, ir.to_proto, ...
import onnx_ir as ir

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# Derive Transformations (allowed to modify the graph) from pattern-based
# rewrite rules
from onnx_passes.passes.base import Transformation, RewriteRulePass

# NumPy used during match condition checks to operate on shapes and tensors
import numpy as np


# Removes all multiplications without effect from the graph, i.e., x * 1 = x
@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-identity")
class EliminateIdentityMul(Transformation, RewriteRulePass):
    @property
    def commute(self):
        return True

    def pattern(self, op, x, a):
        return op.Mul(x, a)

    def check(self, op, x, a):
        if a := ir.convenience.get_const_tensor(a):
            return np.all(a.numpy() == 1)
        return False

    def rewrite(self, op, x, a):
        return op.Identity(x)


# Removes all bitwise-and without effect from the graph, i.e., x & 11...1 = x
@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-identity")
class EliminateIdentityBitwiseAnd(Transformation, RewriteRulePass):
    @property
    def commute(self):
        return True

    def pattern(self, op, x, a):
        return op.BitwiseAnd(x, a)

    def check(self, op, x, a):
        if a := ir.convenience.get_const_tensor(a):
            return np.all(a.numpy() == ~0)
        return False

    def rewrite(self, op, x, a):
        return op.Identity(x)


# Removes all logical-and without effect from the graph, i.e., x and True = x
@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-identity")
class EliminateIdentityAnd(Transformation, RewriteRulePass):
    @property
    def commute(self):
        return True

    def pattern(self, op, x, a):
        return op.And(x, a)

    def check(self, op, x, a):
        if a := ir.convenience.get_const_tensor(a):
            return np.all(a.numpy() == True)
        return False

    def rewrite(self, op, x, a):
        return op.Identity(x)


# Removes all additions without effect from the graph, i.e., x + 0 = x
@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-identity")
class EliminateIdentityAdd(Transformation, RewriteRulePass):
    @property
    def commute(self):
        return True

    def pattern(self, op, x, a):
        return op.Add(x, a)

    def check(self, op, x, a):
        if a := ir.convenience.get_const_tensor(a):
            return np.all(a.numpy() == 0)
        return False

    def rewrite(self, op, x, a):
        return op.Identity(x)


# Removes all bitwise-or without effect from the graph, i.e., x | 0 = x
@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-identity")
class EliminateIdentityBitwiseOr(Transformation, RewriteRulePass):
    @property
    def commute(self):
        return True

    def pattern(self, op, x, a):
        return op.BitwiseOr(x, a)

    def check(self, op, x, a):
        if a := ir.convenience.get_const_tensor(a):
            return np.all(a.numpy() == 0)
        return False

    def rewrite(self, op, x, a):
        return op.Identity(x)


# Removes all logical-or without effect from the graph, i.e., x or False = x
@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-identity")
class EliminateIdentityOr(Transformation, RewriteRulePass):
    @property
    def commute(self):
        return True

    def pattern(self, op, x, a):
        return op.Or(x, a)

    def check(self, op, x, a):
        if a := ir.convenience.get_const_tensor(a):
            return np.all(a.numpy() == False)
        return False

    def rewrite(self, op, x, a):
        return op.Identity(x)


# Removes all bitwise-xor without effect from the graph, i.e., x ^ 0 = x
@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-identity")
class EliminateIdentityBitwiseXor(Transformation, RewriteRulePass):
    @property
    def commute(self):
        return True

    def pattern(self, op, x, a):
        return op.BitwiseXor(x, a)

    def check(self, op, x, a):
        if a := ir.convenience.get_const_tensor(a):
            return np.all(a.numpy() == 0)
        return False

    def rewrite(self, op, x, a):
        return op.Identity(x)


# Removes all logical-xor without effect from the graph, i.e., x != False = x
@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-identity")
class EliminateIdentityXor(Transformation, RewriteRulePass):
    @property
    def commute(self):
        return True

    def pattern(self, op, x, a):
        return op.Xor(x, a)

    def check(self, op, x, a):
        if a := ir.convenience.get_const_tensor(a):
            return np.all(a.numpy() == False)
        return False

    def rewrite(self, op, x, a):
        return op.Identity(x)


# Removes all bit-shifts without effect from the graph, i.e., x << 0 (>> 0) = x
@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-identity")
class EliminateIdentityBitShift(Transformation, RewriteRulePass):
    @property
    def commute(self):
        return True

    def pattern(self, op, x, a):
        return op.BitShift(x, a)

    def check(self, op, x, a):
        if a := ir.convenience.get_const_tensor(a):
            return np.all(a.numpy() == 0)
        return False

    def rewrite(self, op, x, a):
        return op.Identity(x)


# Checks for matrix eye being a valid identity matrix to be multiplied with
# matrix x from either side
def check_identity_matmul(x, eye):
    # Try to unpack the shapes, raises ValueError if there are not enough
    # dimensions to unpack (identity matrix needs at least 2 dimensions)
    try:
        *_, N, M = eye.shape
    except ValueError:
        return False

    # The potential identity matrix must be square and match the last two
    # dimensions of the intput (only last dimension in case of 1D input)
    if N == M and tuple(x.shape[-2:]) in {(N, N), (N,)}:
        if eye := ir.convenience.get_const_tensor(eye):
            # Broadcasts over any batch dimensions
            return np.all(eye == np.eye(N, N))

    # Not constant or not a valid identity matrix
    return False


# Eliminates constant matrix multiplications without effect, i.e.,
# multiplications by the identity matrix which exists for square matrices
@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-identity")
class EliminateIdentityMatMulLhs(Transformation, RewriteRulePass):
    def pattern(self, op, x, eye):
        return op.MatMul(eye, x)

    def check(self, op, x, eye):
        return check_identity_matmul(x, eye)

    def rewrite(self, op, x, eye):
        return op.Identity(x)


# Eliminates constant matrix multiplications without effect, i.e.,
# multiplications by the identity matrix which exists for square matrices
@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-identity")
class EliminateIdentityMatMulRhs(Transformation, RewriteRulePass):
    def pattern(self, op, x, eye):
        return op.MatMul(x, eye)

    def check(self, op, x, eye):
        return check_identity_matmul(x, eye)

    def rewrite(self, op, x, eye):
        return op.Identity(x)


# Eliminates Where operators if the condition is a constant and always chooses
# the same branch: This rule selects the left hand side if possible
@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-where")
class EliminateWhereLhs(Transformation, RewriteRulePass):
    def pattern(self, op, condition, lhs, rhs):
        return op.Where(condition, lhs, rhs)

    def check(self, op, condition, lhs, rhs):
        if condition := ir.convenience.get_const_tensor(condition):
            return np.all(condition.numpy())
        return False

    def rewrite(self, op, condition, lhs, rhs):
        return op.Identity(lhs)


# Eliminates Where operators if the condition is a constant and always chooses
# the same branch: This rule selects the right hand side if possible
@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-where")
class EliminateWhereRhs(Transformation, RewriteRulePass):
    def pattern(self, op, condition, lhs, rhs):
        return op.Where(condition, lhs, rhs)

    def check(self, op, condition, lhs, rhs):
        if condition := ir.convenience.get_const_tensor(condition):
            return np.all(condition.numpy() == False)
        return False

    def rewrite(self, op, condition, lhs, rhs):
        return op.Identity(rhs)


# Eliminates type-casts where the target type is known and the same as the type
# of the input: This rule matches the Cast operator with attribute target type
@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-identity")
class EliminateIdentityCast(Transformation, RewriteRulePass):
    def pattern(self, op, x, to):
        return op.Cast(x, to=to)

    def check(self, op, x, to):
        return x.dtype == to.as_int()

    def rewrite(self, op, x, to):
        return op.Identity(x)


# Eliminates type-casts where the target type is known and the same as the type
# of the input: This rule matches the Cast operator with target type derived
# from a second input
@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-identity")
class EliminateIdentityCastLike(Transformation, RewriteRulePass):
    def pattern(self, op, x, y):
        return op.CastLike(x, y)

    def check(self, op, x, y):
        return x.dtype == y.dtype

    def rewrite(self, op, x, y):
        return op.Identity(x)
