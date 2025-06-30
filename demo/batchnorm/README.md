# Demonstration of Batch Normalization Streamlining
This demo model comprises a single batch normalization operator node, which, in
inference mode, can be inlined as affine scale and bias operations, i.e.,
elementwise multiplication and addition along a single channel axis.
```bash
netron --browse model.onnx
```

## Inlining Batch Normalization without Streamlining
Inlining batch normalization turns the single BatchNorm operator node into four
elementwise binary operations: A Sub subtracting the mean, a Div dividing by the
standard deviation (square root of the variance), a Mul multiplying by the
learned affine scale and an Add adding the learned bias:
```bash
onnx-passes -c cfg.yaml -o out.onnx model.onnx \
 shape-inference inline-batchnorm fold-constants cleanup
netron --browse out.onnx
```

## Streamlined Batch Normalization
Combining inlining with streamlining and constant folding, fuses the two scaling
operations (Div and Mul) and the two bias operations (Sub and Add) into a single
Mul and Add operation each which are then reordered to have Mul followed by Add:
```bash
onnx-passes -c cfg.yaml -o out.onnx model.onnx \
 shape-inference inline-batchnorm streamline fold-constants cleanup
netron --browse out.onnx
```
