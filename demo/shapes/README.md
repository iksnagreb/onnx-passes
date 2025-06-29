# Demonstration of Shape Streamlining
This demo model comprises a sequence of reshape operations (Squeeze, Unsqueeze,
Flatten, Reshape) which effectively have the input shape at the output. All
shapes are constant, either by being specified explicitly or by being derived
from the constant input shape.
```bash
netron --browse model.onnx
```

## Shape Streamlining without Constant-Folding
Shape streamlining tries to express all Reshape-like operations as standardized
Reshape operations with constant shape. However, without constant folding, shape
propagation might be blocked, preventing some streamlining transformations from
being applicable. The resulting graph can be a mess:
```bash
onnx-passes -c cfg.yaml -o out.onnx model.onnx \
 streamline-shapes shape-inference cleanup
netron --browse out.onnx
```

## Shape Streamlining with Constant-Folding
When combining shape streamlining with constant folding, shapes propagate and
the whole graph eventually collapses into an Identity function, just as
expected:
```bash
onnx-passes -c cfg.yaml -o out.onnx model.onnx \
 streamline-shapes shape-inference fold-constants cleanup
netron --browse out.onnx
```
