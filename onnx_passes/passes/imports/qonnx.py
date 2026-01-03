# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# Derive Transformations (allowed to modify the graph) from pattern-based
# rewrite rules
from onnx_passes.passes.base import Pass, FunctionalPass, Transformation, \
    RewriteRulePass

# Domain used by custom operators implemented with this library
from onnx_passes.ops import DOMAIN as CUSTOM_DOMAIN, inject_custom_ops
# Domain used by QONNX operators which are to be transplanted into CUSTOM_DOMAIN
from onnx_passes.ops.qonnx import DOMAIN as QONNX_DOMAIN, BREVITAS_DOMAIN

# Utilities to check properties of values and collecting attributes from ONNX
# nodes and generating matching tensors of ones
from onnx_passes.passes.util import collect_attrs, is_constant, ones_like


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


# Imports QONNX MultiThreshold custom operator from the QONNX domain into the
# CUSTOM_DOMAIN to enable ONNX Runtime execution
class ImportQONNXMultiThreshold(Transformation, RewriteRulePass):
    def pattern(self, op, x, thresholds):
        return op.MultiThreshold(
            x, thresholds, _domain=QONNX_DOMAIN, _outputs=["y"]
        )

    def check(self, op, x, thresholds, y):
        if x.shape is not None and x.shape.is_static():
            return is_constant(thresholds)
        return False

    def rewrite(self, op, x, thresholds, y):
        # Default attributes of the MultiThreshold operator according to QONNX
        # reference
        attributes = {
            "out_dtype": (ir.AttributeType.STRING, ""),
            "out_scale": (ir.AttributeType.FLOAT, 1.0),
            "out_bias": (ir.AttributeType.FLOAT, 0.0),
            "data_layout": (ir.AttributeType.STRING, "NCHW"),
        }

        # Collect node attributes falling back to defaults defined above
        attributes = collect_attrs(y.producer(), attributes)

        # Find the channel dimension (see data layout concept in QONNX...)
        cdim = attributes["data_layout"].as_string().index("C")

        # The thresholds tensor needs to be expanded to explicitly match the
        # dimensions of the input
        shape = [*(len(x.shape) * [1]), thresholds.shape[-1]]
        shape[cdim] = x.shape[cdim]

        # Expand the thresholds input to make all dimensions explicit so we can
        # get rid of tracking layout information
        thresholds = op.Reshape(
            op.Expand(
                thresholds,
                op.Constant(value_ints=[x.shape[cdim], thresholds.shape[-1]])
            ),
            op.Constant(value_ints=shape)
        )

        # QONNX MultiThresholds are always monotonically increasing, i.e.,
        # positive unit steps
        weights = ones_like(op, thresholds)

        # Replacement pattern with explicit weights, scale and bias, inserted
        # into our custom domain
        return op.Add(
            op.Mul(
                op.Constant(value_float=attributes["out_scale"].as_float()),
                op.MultiThreshold(x, thresholds, weights, _domain=CUSTOM_DOMAIN)
            ),
            op.Constant(value_float=attributes["out_bias"].as_float())
        )


# TODO: Import BipolarQuant and Trunc from the QONNX domain...


# ir.Model, ir.passes.PassResult, ir.from_proto, ir.to_proto, ...
import onnx_ir as ir

# Opset version conversion pass build into ONNX Script
from onnxscript.version_converter import ConvertVersionPass

# Minimum opset version required to implement QONNX operators in pure ONNX
QONNX_MINIMUM_OPSET_VERSION = 19


# QONNX Quant function implements configurable rounding mode via string
# comparison inside the graph, Equal supports string comparison since opset 19
class _ConvertQONNXMinimumVersion(Pass, FunctionalPass):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # No need to convert the version if already above the minimum
        onnx_opset_version = model.graph.opset_imports[""]
        if onnx_opset_version >= QONNX_MINIMUM_OPSET_VERSION:
            return ir.passes.PassResult(
                ir.from_proto(ir.to_proto(model)), False
            )

        # Convert to the minimum version required
        result = ConvertVersionPass(QONNX_MINIMUM_OPSET_VERSION)(model)

        # Re-inject the custom operator functions into the models inlines and
        # removes function definitions
        return ir.passes.PassResult(inject_custom_ops(result.model), True)


# Bundles QONNX operator import and required version conversion passes in the
# required order
@passes.register("import-qonnx")
class ImportQONNX(passes.compose.ComposePass, FunctionalPass):
    __passes__ = [
        _ConvertQONNXMinimumVersion,
        ImportQONNXQuant,
        ImportBrevitasQuant,
        ImportQONNXMultiThreshold
    ]
