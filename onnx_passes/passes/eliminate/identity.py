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
        return op.Mul(x, a, _outputs=["_out"])

    def check(self, op, x, a, _out):
        if a := ir.convenience.get_const_tensor(a):
            return _out.shape is not None and np.all(a.numpy() == 1)
        return False

    def rewrite(self, op, x, a, _out):
        return op.Expand(x, op.Constant(value_ints=list(_out.shape)))


# Removes all divisions without effect from the graph, i.e., x / 1 = x
@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-identity")
class EliminateIdentityDiv(Transformation, RewriteRulePass):
    def pattern(self, op, x, a):
        return op.Div(x, a, _outputs=["_out"])

    def check(self, op, x, a, _out):
        if a := ir.convenience.get_const_tensor(a):
            return _out.shape is not None and np.all(a.numpy() == 1)
        return False

    def rewrite(self, op, x, a, _out):
        return op.Expand(x, op.Constant(value_ints=list(_out.shape)))


# Removes all bitwise-and without effect from the graph, i.e., x & 11...1 = x
@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-identity")
class EliminateIdentityBitwiseAnd(Transformation, RewriteRulePass):
    @property
    def commute(self):
        return True

    def pattern(self, op, x, a):
        return op.BitwiseAnd(x, a, _outputs=["_out"])

    def check(self, op, x, a, _out):
        if a := ir.convenience.get_const_tensor(a):
            return _out.shape is not None and np.all(a.numpy() == ~0)
        return False

    def rewrite(self, op, x, a, _out):
        return op.Expand(x, op.Constant(value_ints=list(_out.shape)))


# Removes all logical-and without effect from the graph, i.e., x and True = x
@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-identity")
class EliminateIdentityAnd(Transformation, RewriteRulePass):
    @property
    def commute(self):
        return True

    def pattern(self, op, x, a):
        return op.And(x, a, _outputs=["_out"])

    def check(self, op, x, a, _out):
        if a := ir.convenience.get_const_tensor(a):
            return _out.shape is not None and np.all(a.numpy() == True)
        return False

    def rewrite(self, op, x, a, _out):
        return op.Expand(x, op.Constant(value_ints=list(_out.shape)))


# Removes all additions without effect from the graph, i.e., x + 0 = x
@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-identity")
class EliminateIdentityAdd(Transformation, RewriteRulePass):
    @property
    def commute(self):
        return True

    def pattern(self, op, x, a):
        return op.Add(x, a, _outputs=["_out"])

    def check(self, op, x, a, _out):
        if a := ir.convenience.get_const_tensor(a):
            return _out.shape is not None and np.all(a.numpy() == 0)
        return False

    def rewrite(self, op, x, a, _out):
        return op.Expand(x, op.Constant(value_ints=list(_out.shape)))


# Removes all subtractions without effect from the graph, i.e., x - 0 = x
@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-identity")
class EliminateIdentitySub(Transformation, RewriteRulePass):
    def pattern(self, op, x, a):
        return op.Sub(x, a, _outputs=["_out"])

    def check(self, op, x, a, _out):
        if a := ir.convenience.get_const_tensor(a):
            return _out.shape is not None and np.all(a.numpy() == 0)
        return False

    def rewrite(self, op, x, a, _out):
        return op.Expand(x, op.Constant(value_ints=list(_out.shape)))


# Removes all bitwise-or without effect from the graph, i.e., x | 0 = x
@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-identity")
class EliminateIdentityBitwiseOr(Transformation, RewriteRulePass):
    @property
    def commute(self):
        return True

    def pattern(self, op, x, a):
        return op.BitwiseOr(x, a, _outputs=["_out"])

    def check(self, op, x, a, _out):
        if a := ir.convenience.get_const_tensor(a):
            return _out.shape is not None and np.all(a.numpy() == 0)
        return False

    def rewrite(self, op, x, a, _out):
        return op.Expand(x, op.Constant(value_ints=list(_out.shape)))


# Removes all logical-or without effect from the graph, i.e., x or False = x
@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-identity")
class EliminateIdentityOr(Transformation, RewriteRulePass):
    @property
    def commute(self):
        return True

    def pattern(self, op, x, a):
        return op.Or(x, a, _outputs=["_out"])

    def check(self, op, x, a, _out):
        if a := ir.convenience.get_const_tensor(a):
            return _out.shape is not None and np.all(a.numpy() == False)
        return False

    def rewrite(self, op, x, a, _out):
        return op.Expand(x, op.Constant(value_ints=list(_out.shape)))


# Removes all bitwise-xor without effect from the graph, i.e., x ^ 0 = x
@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-identity")
class EliminateIdentityBitwiseXor(Transformation, RewriteRulePass):
    @property
    def commute(self):
        return True

    def pattern(self, op, x, a):
        return op.BitwiseXor(x, a, _outputs=["_out"])

    def check(self, op, x, a, _out):
        if a := ir.convenience.get_const_tensor(a):
            return _out.shape is not None and np.all(a.numpy() == 0)
        return False

    def rewrite(self, op, x, a, _out):
        return op.Expand(x, op.Constant(value_ints=list(_out.shape)))


# Removes all logical-xor without effect from the graph, i.e., x != False = x
@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-identity")
class EliminateIdentityXor(Transformation, RewriteRulePass):
    @property
    def commute(self):
        return True

    def pattern(self, op, x, a):
        return op.Xor(x, a, _outputs=["_out"])

    def check(self, op, x, a, _out):
        if a := ir.convenience.get_const_tensor(a):
            return _out.shape is not None and np.all(a.numpy() == False)
        return False

    def rewrite(self, op, x, a, _out):
        return op.Expand(x, op.Constant(value_ints=list(_out.shape)))


# Removes all bit-shifts without effect from the graph, i.e., x << 0 (>> 0) = x
@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-identity")
class EliminateIdentityBitShift(Transformation, RewriteRulePass):
    @property
    def commute(self):
        return True

    def pattern(self, op, x, a):
        return op.BitShift(x, a, _outputs=["_out"])

    def check(self, op, x, a, _out):
        if a := ir.convenience.get_const_tensor(a):
            return _out.shape is not None and np.all(a.numpy() == 0)
        return False

    def rewrite(self, op, x, a, _out):
        return op.Expand(x, op.Constant(value_ints=list(_out.shape)))


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
        return op.MatMul(eye, x, _outputs=["_out"])

    def check(self, op, x, eye, _out):
        return _out.shape is not None and check_identity_matmul(x, eye)

    def rewrite(self, op, x, eye, _out):
        return op.Expand(x, op.Constant(value_ints=list(_out.shape)))


# Eliminates constant matrix multiplications without effect, i.e.,
# multiplications by the identity matrix which exists for square matrices
@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-identity")
class EliminateIdentityMatMulRhs(Transformation, RewriteRulePass):
    def pattern(self, op, x, eye):
        return op.MatMul(x, eye, _outputs=["_out"])

    def check(self, op, x, eye, _out):
        return _out.shape is not None and check_identity_matmul(x, eye)

    def rewrite(self, op, x, eye, _out):
        return op.Expand(x, op.Constant(value_ints=list(_out.shape)))


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


# Eliminates Expand (broadcast) where the target shape is known and the same as
# the static shape of the input
@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-identity")
class EliminateIdentityExpand(Transformation, RewriteRulePass):
    def pattern(self, op, x, shape):
        return op.Expand(x, shape)

    def check(self, op, x, shape):
        if x.shape is not None and x.shape.is_static():
            if (shape := ir.convenience.get_const_tensor(shape)) is not None:
                return np.all(shape.numpy() == x.shape)
        return False

    def rewrite(self, op, x, shape):
        return op.Identity(x)


# Domain used by custom operators implemented with this library
from onnx_passes.ops import DOMAIN as CUSTOM_DOMAIN
# Make custom Im2Col operator available for convolution lowering
from onnx_passes.ops.im2col import Im2Col  # noqa: Used indirectly via registry


# Eliminates Im2Col (input generators) where the output shape is the same as the
# input shape, which is given if the kernel, stride and dilation are all 1
@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-identity")
class EliminateIdentityIm2Col(Transformation, RewriteRulePass):
    def pattern(self, op, x, indices, dilations, kernel_shape, strides):
        return op.Im2Col(
            x, indices, dilations=dilations, kernel_shape=kernel_shape,
            strides=strides, _domain=CUSTOM_DOMAIN
        )

    def check(self, op, x, indices, dilations, kernel_shape, strides):
        if dilations is not None:
            if np.any(np.asarray(dilations.as_ints()) != 1):
                return False

        if kernel_shape is not None:
            if np.any(np.asarray(kernel_shape.as_ints()) != 1):
                return False

        if strides is not None:
            if np.any(np.asarray(strides.as_ints()) != 1):
                return False

        return True

    def rewrite(self, op, x, indices, dilations, kernel_shape, strides):
        return op.Identity(x)


# Identity elimination pass build into ONNX IR and ONNXScript
from onnx_ir.passes.common import IdentityEliminationPass


@passes.verify.equality
@passes.register("eliminate")
@passes.register("eliminate-identity")
class EliminateIdentity(passes.base.Transformation):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return IdentityEliminationPass()(model)
