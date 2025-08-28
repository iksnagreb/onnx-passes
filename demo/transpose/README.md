# Demonstration of Transpose Streamlining
This demo model comprises a sequence of transpose operations alongside simple
elementwise operations, such as addition, multiplication and activation
functions such as Relu and Gelu. All but the transpose following the Gelu are
matched by another transpose and the overall permutation of axes could be
expressed by a single transpose at the output.
```bash
netron --browse model.onnx
```

## Transpose Streamlining with Constant-Folding
Streamlining transposes is categorized as shape streamlining. Without constant 
folding, transpose propagation and elimination might be blocked. When combining
transpose and shape streamlining with constant folding, shapes and transposes
propagate and can be folded into constant parameter tensors. As expected, only
the unmatched transpose following the Gelu remain and propagates to the end of
the graph:
```bash
onnx-passes -c cfg.yaml -o out.onnx model.onnx \
 streamline-shapes shape-inference fold-constants cleanup checker verify
netron --browse out.onnx
```
