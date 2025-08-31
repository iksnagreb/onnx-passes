# Datatype annotations ir.DataType.FLOAT
import onnx_ir as ir

# Custom operator function registry
from onnx_passes.ops import register, op
# Custom rounding function with rounding mode configured via string attribute
from onnx_passes.ops.round import Round

# The domain of custom operators exported by QONNX
DOMAIN = "qonnx.custom_op.general"
# Brevitas exports to the brevitas domain, which, however, can be transplanted
# to the QONNX domain
BREVITAS_DOMAIN = "onnx.brevitas"


# QONNX quantizer custom operator implementation to allow models with custom
# quantization to be executed via ONNX Runtime
#   See https://github.com/fastmachinelearning/qonnx for details....
@register
def Quant(x, scale, zeropoint, bitwidth,  # noqa: Operator name is uppercase
          signed: int, narrow: int, rounding_mode: str):
    # Quantizer attributes are specified as integers but are use in calculations
    # together with float inputs - inputs to Add, Mul, etc. must match in type
    signed = op.Cast(signed, to=ir.DataType.FLOAT)
    narrow = op.Cast(narrow, to=ir.DataType.FLOAT)

    # Minimum representable integer of signed bitwidth taking narrow range
    # into account - calculations inlined into the graph, depends on dynamic
    # bitwidth
    _min = (- 2.0 ** (bitwidth - signed) + narrow) * signed

    # Maximum representable integer of signed bitwidth taking narrow range
    # into account - calculations inlined into the graph, depends on dynamic
    # bitwidth
    _max = 2.0 ** (bitwidth - signed) - 1 - narrow * (1 - signed)

    # Scale and zero point: Float to Integer
    q = op.Add(op.Div(x, scale), zeropoint)

    # This simulates if-else branching without an if operator - usually the
    # condition should eventually evaluate to a constant expression allowing
    # one branch to be eliminated. op.Where also takes care of broadcasting.
    q = op.Where(
        # Condition: if bitwidth == 1 and signed - signed 1-bit needs manual
        # fix...
        op.And(
            op.Equal(bitwidth, 1.0), op.Cast(signed, to=ir.DataType.BOOL)
        ),
        # If-branch: Fix 1-bit quantization as manually converted bipolar
        # encoding
        op.Where(
            op.GreaterOrEqual(q, 0.0), op.CastLike(1.0, q), op.CastLike(-1.0, q)
        ),
        # Else-branch: Clip the integer to the range and round according to
        # the rounding mode while ensuring the data type to stay the same
        Round(op.Clip(q, _min, _max), rounding_mode=rounding_mode)
    )

    # Scale and zero point: Integer to Float
    return op.Mul(op.Sub(q, zeropoint), scale)
