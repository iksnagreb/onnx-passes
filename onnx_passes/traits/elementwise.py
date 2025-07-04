# Op types of ONNX operators categorized as elementwise unary, i.e., applying
# the same function to each individual element of the input producing single
# output of the same shape.
ELEMENTWISE_UNARY = {
    "Abs",
    "Acos",
    "Acosh",
    "Asin",
    "Asinh",
    "Atan",
    "Atanh",
    "BitwiseNot",
    "Cast",
    "Ceil",
    "Celu",
    "Cos",
    "Cosh",
    "Elu",
    "Erf",
    "Exp",
    "Floor",
    "Gelu",
    "HardSigmoid",
    "HardSwish",
    "Identity",
    "IfInf",
    "IsNaN",
    "LeakyRelu",
    "Log",
    "Mish",
    "Neg",
    "Not",
    "Reciprocal",
    "Relu",
    "Round",
    "Selu",
    "Shrink",
    "Sigmoid",
    "Sign",
    "Sin",
    "Sinh",
    "Softplus",
    "Softsign",
    "Sqrt",
    "Tan",
    "Tanh",
    "ThresholdedRelu",
}

# Op types of ONNX operators categorized as elementwise binary, i.e., applying
# the same function to pairs of individual elements from each of the inputs
# producing a single output.
ELEMENTWISE_BINARY = {
    "Add",
    "Sub",
    "Mul",
    "Div",
    "BitwiseOr",
    "BitwiseAnd",
    "BitwiseXor",
    "BitShift",
    "Or",
    "And",
    "Xor",
    "Equal",
    "Less",
    "LessOrEqual",
    "Greater",
    "GreaterOrEqual",
    "Mod",
    "Pow",
    "PRelu",
}

# Op types of ONNX operators categorized as elementwise ternary, i.e., applying
# the same function to triples of individual elements from each of the inputs
# producing a single output.
ELEMENTWISE_TERNARY = [
    "Clip",
    "Where"
]

# Op types of ONNX operators categorized as elementwise n-ary, i.e., applying
# the same function to tuples of individual elements from each of the inputs
# producing a single output.
ELEMENTWISE_NARY = {
    # All unary, binary and ternary are n-ary as well
    *ELEMENTWISE_UNARY,
    *ELEMENTWISE_BINARY,
    *ELEMENTWISE_TERNARY,
    # These have variadic inputs, according to the ONNX reference between 1 and
    # 2147483647 inputs. These are elementwise versions, not the reductions, the
    # reduction versions of these operators are called Reduce<Op>.
    "Max",
    "Mean",
    "Min",
    "Sum"
}

# ONNX IR operator node representation
from onnx_ir import Node


def is_elementwise_unary(op: str | Node):
    if isinstance(op, Node):
        return op.op_type in ELEMENTWISE_UNARY
    return op in ELEMENTWISE_UNARY


def is_elementwise_binary(op: str | Node):
    if isinstance(op, Node):
        return op.op_type in ELEMENTWISE_BINARY
    return op in ELEMENTWISE_BINARY


def is_elementwise_ternary(op: str | Node):
    if isinstance(op, Node):
        return op.op_type in ELEMENTWISE_TERNARY
    return op in ELEMENTWISE_TERNARY


def is_elementwise(op: str | Node):
    if isinstance(op, Node):
        return op.op_type in ELEMENTWISE_NARY
    return op in ELEMENTWISE_NARY
