# Demonstration of Importing and Inlining Brevitas Quantizers
This demo model comprises a simple uniform per-tensor quantizer exported from
[Brevitas](https://github.com/Xilinx/brevitas). These quantizers implement a similar quantization schemed as
the standard ONNX Quantize-Dequantize format, but as single, unified operator
instead of two separate, one for each half. The underlying operations of
quantizing and dequantizing the input tensor can be inlined into the graph as a
chain of scaling and rounding operations.
```bash
onnx-passes --browse -o out.onnx --verify.inputs inp.npy model.onnx \
 canonicalize
```

## Inlining and Streamlining Quantizers into the Graph
Once the Quant operators are imported for model verification, they can be
inlined and streamlined which results in a model equivalently implementing the
quantization operation as a chain of scaling, clipping, rounding and scaling
operations from standard ONNX.
```bash
onnx-passes --browse -o out.onnx --verify.inputs inp.npy model.onnx \
 canonicalize inline.quantizer reorder
```
