# Quantized Layer Tail to Thresholds Demonstration
This demo model comprises a simple so-called layer tail: a
bias-batchnorm-activation-quantizer chain which can be inlined and streamlined
into the graph (see https://arxiv.org/abs/2508.21493)
The demo model is quantized and exported via
[Brevitas](https://github.com/Xilinx/brevitas) and already imported into the
custom opset domain `onnx_passes.ops` to make the model executable via ONNX
Runtime, see the [Quantizer demo](../quant/README.md) for details.
```bash
netron --browse model.onnx
```

## Inlining Quantizers and Batch Normalization into the Graph
Once the Quant operators can be inlined which results in a model equivalently
implementing the quantization operation as a chain of scaling, clipping,
rounding and scaling operations from the standard ONNX opset. Likewise, batch
normalization can be inlined as affine scale and bias operations, i.e.,
elementwise multiplication and addition along a single channel axis. This leaves
us with the fully expanded layer tail graph in all its glory:
```bash
onnx-passes -c cfg.yaml -o out.onnx model.onnx inline-qonnx inline-batchnorm shape-inference checker verify
netron --browse out.onnx
```

## Threshold Conversion and Streamlining the Graph
After inlining all operators, the graph can be gradually optimized and
simplified by streamlining: Operators can be reordered and like terms can be
fused until finally rounding operations (and thus quantization) can be expressed
as multi-threshold functions. These multi-threshold functions - which are 
comparisons at the core - can absorb elementwise (monotonic) functions, such as
addition, (positive) multiplication or the Sigmoid activation This fuses almost
the whole layer tail into a single operator. At the output the dequantization
scale and bias (usually only present for signed quantizers) remain, which, in a
complete model, would be streamlinable into the next layer.
```bash
onnx-passes -c cfg.yaml -o out.onnx model.onnx inline-qonnx inline-batchnorm streamline-thresholds streamline checker verify
netron --browse out.onnx
```
