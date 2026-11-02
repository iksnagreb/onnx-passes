# Demonstration of Shape Streamlining
This demo model comprises a sequence of reshape operations (Squeeze, Unsqueeze,
Flatten, Reshape) which effectively have the input shape at the output. All
shapes are constant, either by being specified explicitly or by being derived
from the constant input shape.
```bash
onnx-passes --browse -o out.onnx --verify.inputs inp.npy model.onnx \
 canonicalize
```

## Shape Streamlining with Constant-Folding
Shape streamlining tries to express all Reshape-like operations as standardized
Reshape operations with constant shape. When combining shape streamlining with
constant folding, shapes propagate and the whole graph eventually collapses into
an Identity function:
```bash
onnx-passes --browse -o out.onnx --verify.inputs inp.npy model.onnx \
 canonicalize reorder
```
