# Demonstration of Transpose Streamlining
This demo model comprises a sequence of transpose operations alongside simple
elementwise operations, such as addition, multiplication and activation
functions such as Relu and Gelu. All but the transpose following the Gelu are
matched by another transpose and the overall permutation of axes could be
expressed by a single transpose at the input (or output).
```bash
onnx-passes --browse -o out.onnx --verify.inputs inp.npy model.onnx \
 canonicalize
```

## Transpose Streamlining with Constant-Folding
Streamlining transposes is part of the reordering passes, where 
elementwise-transpose are reordered such that transposes propagate upwards the graph and combine cancel out via normalization passes.
```bash
onnx-passes --browse -o out.onnx --verify.inputs inp.npy model.onnx \
 canonicalize reorder
```
