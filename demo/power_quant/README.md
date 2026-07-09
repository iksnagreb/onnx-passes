Attempts at rewriting the PowerQuant (see https://arxiv.org/abs/2301.09858)
quantized matrix multiplication pattern for implementation in FINN.

# Export

```bash
python export.py
netron --browse model.onnx
```

# Cleanup

```bash
onnx-passes -c cfg.yaml -o cleaned.onnx model.onnx \
  shape-inference cleanup checker verify
netron --browse cleaned.onnx
```

# Extract and Fuse

```bash
onnx-passes -c cfg.yaml -o fused.onnx model.onnx \
  shape-inference cleanup FusePowerQuant link-ops shape-inference \
  fold-constants checker verify
netron --browse fused.onnx
```
