# ir.Model, ir.passes.PassResult, ir.from_proto, ir.to_proto, ...
import onnx_ir as ir

# Opset version conversion pass build into ONNX Script
from onnxscript.version_converter import ConvertVersionPass

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes


# TODO: Instantiating all these manually is slightly annoying... but as long as
#  each simply calls to the built-in ConvertVersionPass it is probably fine...


# Converts the model to ONNX opset version 18 if supported
@passes.verify.equality
@passes.register("convert-opset18")
class ConvertVersion18(passes.base.Annotation):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return ConvertVersionPass(18)(ir.from_proto(ir.to_proto(model)))


# Converts the model to ONNX opset version 19 if supported
@passes.verify.equality
@passes.register("convert-opset19")
class ConvertVersion19(passes.base.Annotation):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return ConvertVersionPass(19)(ir.from_proto(ir.to_proto(model)))


# Converts the model to ONNX opset version 20 if supported
@passes.verify.equality
@passes.register("convert-opset20")
class ConvertVersion20(passes.base.Annotation):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return ConvertVersionPass(20)(ir.from_proto(ir.to_proto(model)))


# Converts the model to ONNX opset version 21 if supported
@passes.verify.equality
@passes.register("convert-opset21")
class ConvertVersion21(passes.base.Annotation):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return ConvertVersionPass(21)(ir.from_proto(ir.to_proto(model)))


# Converts the model to ONNX opset version 22 if supported
@passes.verify.equality
@passes.register("convert-opset22")
class ConvertVersion22(passes.base.Annotation):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return ConvertVersionPass(22)(ir.from_proto(ir.to_proto(model)))


# Converts the model to ONNX opset version 23 if supported
@passes.verify.equality
@passes.register("convert-opset23")
class ConvertVersion23(passes.base.Annotation):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return ConvertVersionPass(23)(ir.from_proto(ir.to_proto(model)))
