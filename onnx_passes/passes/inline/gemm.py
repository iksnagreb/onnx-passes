# ir.Value, ir.Attr, ir.tensor
import onnx_ir as ir

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# Derive Transformations (allowed to modify the graph) from pattern-based
# rewrite rules
from onnx_passes.passes.base import Transformation, RewriteRuleSetPass

# Collecting node attributes with optional defaults
from onnx_passes.passes.util import collect_attrs


# Inlines General Matrix Multiplication (Gemm) operators into the graph as a
# pattern of MatMul, Add and Transposes. Matches a set of two patterns as the
# addition is optional.
@passes.verify.tolerance
@passes.register("inline-gemm")
class InlineGemm(Transformation, RewriteRuleSetPass):
    # Default attributes of the Gemm operator according to ONNX operators
    # reference: https://onnx.ai/onnx/operators/onnx__Gemm.html
    ATTRIBUTES = {
        "alpha": (ir.AttributeType.FLOAT, 1.0),
        "beta": (ir.AttributeType.FLOAT, 1.0),
        "transA": (ir.AttributeType.INT, 0),
        "transB": (ir.AttributeType.INT, 0),
    }

    def pattern(self):
        return [
            lambda op, a, b, c: op.Gemm(a, b, c, _outputs=["y"]),
            lambda op, a, b: op.Gemm(a, b, _outputs=["y"]),
        ]

    def check(self):
        return [
            lambda *args, **kwargs: True,
            lambda *args, **kwargs: True,
        ]

    def rewrite(self):
        # Replacement handling both pattern alternatives depending on whether
        # the optional input c has been matched
        def _rewrite(op, a, b, y, c=None):
            # Collect node attributes falling back to defaults defined above
            attributes = collect_attrs(y.producer(), InlineGemm.ATTRIBUTES)

            # If enabled by attribute, transpose the inputs a and/or b
            a = [a, op.Transpose(a)][attributes["transA"].as_int()]
            b = [b, op.Transpose(b)][attributes["transB"].as_int()]

            # Convert alpha and beta attributes to constant tensors which can be
            # used as inputs to operators
            alpha = op.Constant(value_float=attributes["alpha"].as_float())
            beta = op.Constant(value_float=attributes["beta"].as_float())

            # If the optional input is not present, insert only the partial
            # pattern, there is no reason to insert addition of matching zeros
            if c is None:
                return op.MatMul(op.Mul(alpha, a), b)
            # Full pattern: MatMul + Add
            return op.Add(op.MatMul(op.Mul(alpha, a), b), op.Mul(beta, c))

        return [
            lambda op, a, b, c, y: _rewrite(op, a, b, y, c),
            lambda op, a, b, y: _rewrite(op, a, b, y),
        ]
