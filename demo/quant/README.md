# Demonstration of Importing and Inlining Brevitas Quantizers
This demo model comprises a simple uniform per-tensor quantizer exported from
[Brevitas](...). These quantizers implement a similar quantization schemed as
the standard ONNX Quantize-Dequantize format, but as single, unified operator
instead of two separate, one for each half. Anyway, the underlying operations of
quantizing and dequantizing the input tensor can be inlined into the graph as a
chain of scaling and rounding operations.
```bash
netron --browse model.onnx
```

## Importing into the Custom Opset Domain
To make the model executable via ONNX Runtime, the custom Quant operator needs
to be imported into the opset domain `ai.onnx.contrib` which is the one
registered via onnxruntime-extensions. This step is necessary and must be done
first, to enable verification of the model after applying passes.
```bash
onnx-passes -c cfg.yaml -o out.onnx model.onnx import-qonnx checker verify
netron --browse out.onnx
```

## Inlining and Streamlining Quantizers into the Graph
Once the Quant operators are imported for model verification, they can be
inlined and streamlined which results in a model equivalently implementing the
quantization operation as a chain of scaling, clipping, rounding and scaling
operations from the standard ONNX opset.
```bash
onnx-passes -c cfg.yaml -o out.onnx model.onnx import-qonnx \
 inline-qonnx shape-inference fold-constants streamline eliminate cleanup
netron --browse out.onnx
```
