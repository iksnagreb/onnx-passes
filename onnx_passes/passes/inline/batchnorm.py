# ir.Value, ir.Attr, ir.tensor
import onnx_ir as ir

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# Derive Transformations (allowed to modify the graph) from pattern-based
# rewrite rules
from onnx_passes.passes.base import Transformation, RewriteRulePass


# Inlines batch-normalization into the graph. Only applies to non-training mode
# batch-normalization as training mode batch-normalization has more complex
# input/output pattern while not offering much optimization potential anyway...
@passes.verify.tolerance
@passes.register("inline-batchnorm")
class InlineBatchNormalization(Transformation, RewriteRulePass):
    def pattern(self, op, x, scale, b, mean, var):
        return op.BatchNormalization(x, scale, b, mean, var, _outputs=["y"]),

    def check(self, op, y, **kwargs):
        if training := y.producer().attributes.get("training_mode", None):
            return training.as_int() == 0
        return True

    def rewrite(self, op, x, scale, b, mean, var, y):
        # Default epsilon according to ONNX operators reference documentation:
        #   https://onnx.ai/onnx/operators/onnx__BatchNormalization.html
        if not (epsilon := y.producer().attributes.get("epsilon", None)):
            epsilon = ir.Attr("epsilon", ir.AttributeType.FLOAT, 1e-05)

        # Expand shapes of parameters (which are along the channel axis=1) to
        # match the input shape for broadcasting.
        shape = [size if i == 1 else 1 for i, size in enumerate(x.shape)]

        # Expand all tensor by injecting reshape operators with constant shapes
        # Note: These should disappear after another round of constant folding
        s = op.Reshape(scale, op.Constant(value_ints=shape))
        b = op.Reshape(b, op.Constant(value_ints=shape))
        m = op.Reshape(mean, op.Constant(value_ints=shape))
        v = op.Reshape(var, op.Constant(value_ints=shape))
        e = op.Constant(value_float=epsilon.as_float())

        # Verbose for: y = (x -  mean) / sqrt(var + epsilon) * scale + b
        return op.Add(op.Mul(op.Div(op.Sub(x, m), op.Sqrt(op.Add(v, e))), s), b)
