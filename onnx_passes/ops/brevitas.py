from onnx_passes.ops.qonnx import Quant_v1 as QONNXQuant_v1

DOMAIN = "onnx.brevitas"


class Quant_v1(QONNXQuant_v1, domain=DOMAIN):
    """QONNX arbitrary bitwidth integer quantization custom operator.

    onnx.brevitas domain wrapper of the operator: Some versions of Brevitas
    export quantizers in this domain instead of qonnx.custom_op.general

    See https://github.com/fastmachinelearning/qonnx for details.
    """
