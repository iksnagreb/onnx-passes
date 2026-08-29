# ir.Attr and ir.AttributeType
import onnx_ir as ir

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# Derive Transformations (allowed to modify the graph) from pattern-based
# rewrite rule sets
from onnx_passes.passes.base import Transformation, RewriteRuleSetPass

# Collecting node attributes with optional defaults
from onnx_passes.passes.util import collect_attrs


# Inlines ReduceLogSumExp to Log(ReduceSum(Exp(x))).
@passes.verify.tolerance
@passes.register("inline-reduce-log-sum-exp")
class InlineReduceLogSumExp(Transformation, RewriteRuleSetPass):
    def pattern(self):
        # ReduceLogSumExp optionally receives axes as a second input.
        return [
            lambda op, x, axes: op.ReduceLogSumExp(x, axes, _outputs=["y"]),
            lambda op, x: op.ReduceLogSumExp(x, _outputs=["y"]),
        ]

    def rewrite(self):
        def _rewrite(op, x, y, axes=None):
            # Defaults according to ONNX operators reference documentation:
            #   https://onnx.ai/onnx/operators/onnx__ReduceLogSumExp.html
            attributes = collect_attrs(
                y.producer(),
                {
                    "keepdims": (ir.AttributeType.INT, 1),
                    "noop_with_empty_axes": (ir.AttributeType.INT, 0),
                },
            )

            kwargs = {
                "keepdims": attributes["keepdims"].as_int(),
                "noop_with_empty_axes": (attributes["noop_with_empty_axes"].as_int()),
            }

            if axes is None:
                return op.Log(op.ReduceSum(op.Exp(x), **kwargs))
            return op.Log(op.ReduceSum(op.Exp(x), axes, **kwargs))

        return [
            lambda op, x, axes, y: _rewrite(op, x, y, axes),
            lambda op, x, y: _rewrite(op, x, y),
        ]
