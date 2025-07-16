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

# NumPy used during match condition checks to operate on shapes and constant
# tensors or attributes
import numpy as np


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

    # Do not apply this to all constant inputs: constant folding version below
    def check(self, op, x, scale, zeropoint, bitwidth, signed, narrow, mode):
        for value in {x, scale, zeropoint, bitwidth}:
            if ir.convenience.get_const_tensor(value) is None:
                return True
        return False

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
        # lot if "normal" expressions and literals were allowed...

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


# Inlines QONNX Quant custom operator nodes from the CUSTOM_DOMAIN into the
# graph as a pattern of standard ONNX operators - constant folds the integer
# part, i.e., up to the rounding function if applicable.
@passes.verify.equality
@passes.register("inline-qonnx")
class FoldConstantQONNXQuant(Transformation, RewriteRulePass):
    def pattern(self, op, x, scale, zeropoint, bitwidth, signed, narrow, mode):
        return op.Quant(
            x, scale, zeropoint, bitwidth, signed=signed, narrow=narrow,
            rounding_mode=mode, _domain=CUSTOM_DOMAIN
        )

    def check(self, op, x, scale, zeropoint, bitwidth, signed, narrow, mode):
        for value in {x, scale, zeropoint, bitwidth}:
            if ir.convenience.get_const_tensor(value) is None:
                return False
        return True

    def rewrite(self, op, x, scale, zeropoint, bitwidth, signed, narrow, mode):
        # Signedness and narrow range attributes are integers but are required
        # as floats for calculations below
        signed = float(signed.as_int())
        narrow = float(narrow.as_int())

        # Get the actual string out of the attribute
        rounding_mode = mode.as_string()

        # Get the constant inputs as numpy arrays
        x = ir.convenience.get_const_tensor(x).numpy()
        scale = ir.convenience.get_const_tensor(scale).numpy()
        zeropoint = ir.convenience.get_const_tensor(zeropoint).numpy()
        bitwidth = ir.convenience.get_const_tensor(bitwidth).numpy()

        # Resolve rounding modes from string identifiers
        rounding_fxs = {
            "ROUND": np.round, "CEIL": np.ceil, "FLOOR": np.floor,
            "ROUND_TO_ZERO": lambda v: np.sign(v) * np.floor(np.abs(v))
        }

        # Scale and zero point: Float to Integer
        q = (x / scale) + zeropoint  # noqa: Duplicate of onnx_passes.ops.qonnx

        # Encode signed 1 bit quantization as bipolar values
        if bitwidth == 1 and signed:
            q = np.where(q >= 0, +1, -1)
        # For all bitwidth larger than 1 clip and round the integer to the range
        # of valid values
        else:
            # Minimum and maximum integer value for the bitwidth, signedness and
            # narrow range combination
            _min = signed * (- 2 ** (bitwidth - signed) + narrow)
            _max = + 2 ** (bitwidth - signed) - 1 - narrow * (1 - signed)
            # Clip the integer to the range and round according tot eh rounding
            # mode while ensuring the data type to stay the same
            q = rounding_fxs[rounding_mode](
                np.clip(q, _min, _max, dtype=q.dtype))

        # Convert numpy arrays back to ONNX constants
        q = op.Constant(value=ir.tensor(q))
        zeropoint = op.Constant(value=ir.tensor(zeropoint))
        scale = op.Constant(value=ir.tensor(scale))

        # Scale and zero point: Integer to Float
        return op.Mul(op.Sub(q, zeropoint), scale)
