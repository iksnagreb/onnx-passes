# Demonstration of Residual Streamlining
This demo model comprises a simple residual block: an empty skipping branch
skipping past a single linear layer, i.e., a MatMul followed by channel-wise
bias Add. To slightly complicate things, a scalar multiplication is added in
front of the block, before the initial fork.
```bash
netron --browse model.onnx
```

## Streamlining Mul past Fork, MatMul and Join
The model can be simplified by moving both the elementwise scale and bias past
the residual block. This is done by exploiting algebraic properties of MatMul,
Mul and Add, namely associativity (reorder the Adds), commutativity (swap Mul
and MatMul) and distributivity (swap Mul and joining Add).
```bash
onnx-passes -c cfg.yaml -o out.onnx model.onnx shape-inference streamline
netron --browse out.onnx
```
