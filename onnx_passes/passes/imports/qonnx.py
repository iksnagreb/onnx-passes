# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# Derive Transformations (allowed to modify the graph) from pattern-based
# rewrite rules
from onnx_passes.passes.base import Transformation, RewriteRulePass

# Domain used by custom operators implemented with this library
from onnx_passes.ops import DOMAIN as CUSTOM_DOMAIN
# Domain used by QONNX operators which are to be transplanted into CUSTOM_DOMAIN
from onnx_passes.ops.qonnx import DOMAIN as QONNX_DOMAIN, BREVITAS_DOMAIN


# Imports QONNX Quant custom operator nodes from the QONNX domain into the
# CUSTOM_DOMAIN to enable ONNX Runtime execution
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

# Imports Brevitas Quant custom operator nodes from the Brevitas domain into the
# CUSTOM_DOMAIN to enable ONNX Runtime execution: Brevitas is closely related to
# QONNX
@passes.register("import-qonnx")
class ImportBrevitasQuant(Transformation, RewriteRulePass):
    def pattern(self, op, x, scale, zeropoint, bitwidth, signed, narrow, mode):
        return op.Quant(
            x, scale, zeropoint, bitwidth, signed=signed, narrow=narrow,
            rounding_mode=mode, _domain=BREVITAS_DOMAIN
        )

    def rewrite(self, op, x, scale, zeropoint, bitwidth, signed, narrow, mode):
        return op.Quant(
            x, scale, zeropoint, bitwidth, signed=signed, narrow=narrow,
            rounding_mode=mode, _domain=CUSTOM_DOMAIN
        )


# TODO: Import BipolarQuant, Trunc and MultiThreshold from the QONNX domain...
