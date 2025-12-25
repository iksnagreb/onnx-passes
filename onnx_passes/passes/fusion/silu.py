# All algebraic passes are transformations derived from pattern-based rewrite
# rules
from onnx_passes.passes.base import Transformation, RewriteRulePass

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# Fused Silu is defined in the custom domain and needs to be made available as
# an ONNX Script function once used
from onnx_passes.ops import DOMAIN as CUSTOM_DOMAIN
from onnx_passes.ops.swish import Silu  # noqa: Used via registry


@passes.verify.equality
@passes.register("fuse-silu")
class FuseSilu(Transformation, RewriteRulePass):
    @property
    def commute(self):
        return True

    def pattern(self, op, x):
        return op.Mul(x, op.Sigmoid(x))

    def rewrite(self, op, x):
        return op.Silu(x, _domain=CUSTOM_DOMAIN)
