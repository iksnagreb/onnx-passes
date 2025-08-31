# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# Derive Transformations (allowed to modify the graph) from pattern-based
# rewrite rules
from onnx_passes.passes.base import Transformation, RewriteRulePass

# Domain used by custom operators implemented with this library
from onnx_passes.ops import DOMAIN as CUSTOM_DOMAIN, inject_custom_ops
# Domain used by QONNX operators which are to be transplanted into CUSTOM_DOMAIN
from onnx_passes.ops.qonnx import DOMAIN as QONNX_DOMAIN, BREVITAS_DOMAIN


# Imports QONNX Quant custom operator nodes from the QONNX domain into the
# CUSTOM_DOMAIN to enable ONNX Runtime execution
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


# ir.Model, ir.passes.PassResult, ir.from_proto, ir.to_proto, ...
import onnx_ir as ir

# Opset version conversion pass build into ONNX Script
from onnxscript.version_converter import ConvertVersionPass

# Minimum opset version required to implement QONNX operators in pure ONNX
QONNX_MINIMUM_OPSET_VERSION = 19


# QONNX Quant function implements configurable rounding mode via string
# comparison inside the graph, Equal supports string comparison since opset 19
class _ConvertQONNXMinimumVersion(passes.base.Annotation):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Always start with a copy of the model as all passes are not in-place
        model = ir.from_proto(ir.to_proto(model))

        # No need to convert the version if already above the minimum
        onnx_opset_version = model.graph.opset_imports[""]
        if onnx_opset_version >= QONNX_MINIMUM_OPSET_VERSION:
            return ir.passes.PassResult(model, False)

        # Convert to the minimum version required
        result = ConvertVersionPass(QONNX_MINIMUM_OPSET_VERSION)(
            ir.from_proto(ir.to_proto(model))
        )

        # Re-inject the custom operator functions into the models inlines and
        # removes function definitions
        return ir.passes.PassResult(inject_custom_ops(result.model), True)


# Bundles QONNX operator import and required version conversion passes in the
# required order
@passes.register("import-qonnx")
class ImportQONNX(passes.compose.ComposePass, passes.base.Transformation):
    __passes__ = [
        _ConvertQONNXMinimumVersion, ImportQONNXQuant, ImportBrevitasQuant
    ]
