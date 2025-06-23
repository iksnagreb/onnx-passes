# ir.Value, ir.Attr, ir.tensor
import onnx_ir as ir

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# Derive Transformations (allowed to modify the graph) from pattern-based
# rewrite rules
from onnx_passes.passes.base import Transformation, RewriteRulePass

# Domain used by custom operators implemented with this library
from onnx_passes.ops import DOMAIN as CUSTOM_DOMAIN
# Domain used by QONNX operators which are to be transplanted into CUSTOM_DOMAIN
from onnx_passes.ops.qonnx import DOMAIN as QONNX_DOMAIN


# Imports QONNX Quant custom operator nodes from the QONNX domain into the
# CUSTOM_DOMAIN to enable ONNX Runtime execution
@passes.register("import")
@passes.register("import-qonnx")
class ImportQONNXQuant(Transformation, RewriteRulePass):
    def pattern(self, op, x, scale, zeropoint, bitwidth, signed, narrow, mode):
        return op.Quant(
            x, scale, zeropoint, bitwidth, signed=signed, narrow=narrow,
            rounding_mode=mode, _domain=QONNX_DOMAIN
        )

    def rewrite(self, op, x, scale, zeropoint, bitwidth, signed, narrow, mode):
        return op.Quant(
            x, scale, zeropoint, bitwidth, signed=signed, narrow=narrow,
            rounding_mode=mode, _domain=CUSTOM_DOMAIN
        )


# TODO: Import BipolarQuant, Trunc and MultiThreshold from the QONNX domain...


# Inlines QONNX Quant custom operator nodes from the CUSTOM_DOMAIN into the
# graph as a pattern of standard ONNX operators
# TODO: Find a mechanism to ensure calling ImportQONNXQuant first, otherwise
#  there is no ONNX Runtime support for the Quant node required to verify...
# TODO: Consider not verifying this transformation to directly allow inlining
#  form the QONNX domain...?
@passes.verify.equality
@passes.register("inline-qonnx")
class InlineQONNXQuant(Transformation, RewriteRulePass):
    def pattern(self, op, x, scale, zeropoint, bitwidth, signed, narrow, mode):
        return op.Quant(
            x, scale, zeropoint, bitwidth, signed=signed, narrow=narrow,
            rounding_mode=mode, _domain=CUSTOM_DOMAIN
        )

    def rewrite(self, op, x, scale, zeropoint, bitwidth, signed, narrow, mode):
        # Signedness and narrow range attributes are integers but are
        # required as constant float tensors for calculations below
        signed = op.Constant(value=ir.tensor(float(signed.as_int())))
        narrow = op.Constant(value=ir.tensor(float(narrow.as_int())))

        # Get the actual string out of the attribute
        rounding_mode = mode.as_string()

        # Some constants for convenience...
        _0 = op.Constant(value=ir.tensor(+0.0))
        _1 = op.Constant(value=ir.tensor(+1.0))
        m1 = op.Constant(value=ir.tensor(-1.0))
        _2 = op.Constant(value=ir.tensor(+2.0))

        # Resolve rounding modes from string identifiers to operator functions
        # within the op rewrite context
        rounding_fxs = {
            "ROUND": op.Round, "CEIL": op.Ceil, "FLOOR": op.Floor,
            "ROUND_TO_ZERO": lambda v: op.Mul(op.Sign(v), op.Floor(op.Abs(v)))
        }

        # Minimum representable integer of signed bitwidth taking narrow range
        # into account - calculations inlined into the graph
        #   Reads as: (- 2 ** (bitwidth - signed) + narrow) * signed
        _min = op.Mul(
            op.Add(op.Neg(op.Pow(_2, op.Sub(bitwidth, signed))), narrow), signed
        )

        # Maximum representable integer of signed bitwidth taking narrow range
        # into account - calculations inlined into the graph
        #   Reads as: + 2 ** (bitwidth - signed) - 1 - narrow * (1 - signed)
        _max = op.Sub(
            op.Sub(op.Pow(_2, op.Sub(bitwidth, signed)), _1),
            op.Mul(narrow, op.Sub(_1, signed))
        )

        # Beginning of the actual pattern to be inserted into the graph - this
        # is all rather verbose and difficult to read... could be simplified a
        # lot if "normal" expressions and literal were allowed...

        # Scale and zero point: Float to Integer
        q = op.Add(op.Div(x, scale), zeropoint)

        # This simulates if-else branching without an if operator - usually the
        # condition should eventually evaluate to a constant expression allowing
        # one branch to be eliminated
        q = op.Where(
            # Condition: if bitwidth == 1 and signed - signed 1-bit needs manual
            # fix...
            op.And(
                op.Equal(bitwidth, _1), op.Cast(signed, to=ir.DataType.BOOL)
            ),
            # If-branch: Fix 1-bit quantization as manually converted bipolar
            # encoding
            op.Where(
                op.GreaterOrEqual(q, _0), op.CastLike(_1, q), op.CastLike(m1, q)
            ),
            # Else-branch: Clip the integer to the range and round according to
            # the rounding mode while ensuring the data type to stay the same
            rounding_fxs[rounding_mode](op.CastLike(op.Clip(q, _min, _max), q))
        )

        # Scale and zero point: Integer to Float
        return op.Mul(op.Sub(q, zeropoint), scale)
