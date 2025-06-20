# Export the Demo Model
```bash
python export.py
```
## Show the Exported Model
```bash
netron --browse model.onnx
```

# Run Demo Sequence of Passes on the Exported Model
```bash
onnx-passes -c cfg.yaml model.onnx -o out.onnx cleanup shape-inference verify
```
## Show the Modified Model
```bash
netron --browse out.onnx
```
