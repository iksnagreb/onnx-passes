# Include basic inlining transformations
import onnx_passes.passes.inline.functions
import onnx_passes.passes.inline.batchnorm
import onnx_passes.passes.inline.gemm
import onnx_passes.passes.inline.softmax
import onnx_passes.passes.inline.log_softmax
import onnx_passes.passes.inline.reduce_log_sum_exp
