# Demonstration of Batch Normalization Streamlining
This demo model comprises a single batch normalization operator node, which, in
inference mode, can be inlined as affine scale and bias operations, i.e.,
elementwise multiplication and addition along a single channel axis.
```bash
onnx-passes --browse -o out.onnx --verify.inputs inp.npy model.onnx \
 canonicalize
```

## Inlining Batch Normalization without Streamlining
Inlining batch normalization turns the single BatchNorm operator node into four
elementwise binary operations: A Sub subtracting the mean, a Div dividing by the
standard deviation (square root of the variance), a Mul multiplying by the
learned affine scale and an Add adding the learned bias:
```bash
onnx-passes --browse -o out.onnx --verify.inputs inp.npy model.onnx \
 canonicalize lower.batchnorm
```

## Streamlined Batch Normalization
Combining inlining with reordering and constant folding, fuses the two scaling
operations (Div and Mul) and the two bias operations (Sub and Add) into a single
Mul and Add operation:
```bash
onnx-passes --browse -o out.onnx --verify.inputs inp.npy model.onnx \
 canonicalize lower.batchnorm reorder
```
