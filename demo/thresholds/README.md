# Quantized Layer Tail to Thresholds Demonstration
This demo model comprises a simple so-called layer tail: a
bias-batchnorm-activation-quantizer chain which can be inlined and streamlined
into the graph.
The demo model is quantized and exported via
[Brevitas](https://github.com/Xilinx/brevitas) and already imported into the
custom opset domain `onnx_passes.ops` to make the model executable via ONNX
Runtime, see the [Quantizer demo](../quant/README.md) for details.
```bash
onnx-passes --browse -o out.onnx --verify.inputs inp.npy model.onnx \
 canonicalize
```

See the following publications for details on streamlining and compilation of
multi-threshold functions, and [FINN](https://github.com/Xilinx/finn)
and [FINN+](https://github.com/eki-project/finn-plus) for a custom hardware
implementation of these operators:
- https://arxiv.org/abs/1709.04060
- https://dl.acm.org/doi/full/10.1145/3807510
- https://arxiv.org/abs/2610.21825

## Inlining Quantizers and Batch Normalization into the Graph
Once the Quant operators can be inlined which results in a model equivalently
implementing the quantization operation as a chain of scaling, clipping,
rounding and scaling operations from the standard ONNX opset. Likewise, batch
normalization can be inlined as affine scale and bias operations, i.e.,
elementwise multiplication and addition along a single channel axis. This leaves
us with the fully expanded layer tail graph in all its glory:
```bash
onnx-passes --browse -o out.onnx --verify.inputs inp.npy model.onnx \
 canonicalize inline.quantizer lower.batchnorm
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
onnx-passes --browse -o out.onnx --verify.inputs inp.npy model.onnx \
 canonicalize inline.quantizer lower.batchnorm multithreshold
```

## Multi-Threshold Decomposition
To target the more limited MultiThreshold backend in FINN, which only supports
monotonically increasing unit steps, multi-threshold functions are decomposed:
```bash
onnx-passes --browse -o out.onnx --verify.inputs inp.npy model.onnx \
 canonicalize inline.quantizer lower.batchnorm multithreshold       \
 multithreshold.decompose unbroadcast
```