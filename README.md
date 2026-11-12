# ONNX Passes
Collection of [ONNX](https://github.com/onnx/onnx) model graph transformation
and analysis passes based on [ONNX IR](https://github.com/onnx/ir-py) and
[ONNX Script](https://github.com/microsoft/onnxscript) for fast and
interactive prototyping of model optimizations

## Motivation
Custom ONNX compilers, such as [FINN](https://github.com/Xilinx/finn) make heavy
use of graph transformations to gradually convert and optimize the ONNX graph to
a representation understood by the compiler backend. The most prominent set of
such graph transformations in the FINN framework is the so-called streamlining
to uncover the underlying integer-only compute of a quantized neural network as
part of the optimizing compiler frontend.

Usually these transformations follow a stereotypical structure of (1) detecting
a particular pattern within the graph, (2) checking a set of conditions on this
pattern and, based on the result of these checks, (3) inserting a new or
modified pattern into the graph. However, most of the work needs to be done by
hand with only thin wrappers around the raw ONNX representation based on
[QONNX](https://github.com/fastmachinelearning/qonnx) This causes significant
development overheads when integrating complex transformations. Furthermore,
this can lead to code-bloat, is error-prone and prevents effective reuse of
transformations outside the FINN and QONNX ecosystem.

This is an experimental reimagination of this graph transformation framework
based on infrastructure provided by ONNX IR and ONNX Script, in particular
pattern-based graph rewriter, to simplify integrating novel graph
transformations by avoiding most of the boilerplate. This pattern-based graph
rewriter closely follows the familiar (1)-(3) process while hiding most of the
graph internals, such as node removal and insertion or wiring node inputs and
outputs to other nodes.

See the [overview poster](docs/finn_workshop_fpl26_poster.pdf) presented at the
FINN workshop at FPL'26 for more context and a few motivational examples.

## Installation
```
pip install onnx-passes
```

## Tightly Verified Transformations
When successively transforming and optimizing a model, it is important to ensure
that the resulting model still matches the original model at least within some
reasonable tolerance.

Via the [ONNX ReferenceEvaluator](https://onnx.ai/onnx/api/reference.html) the
model can be executed after each individual pass application, allowing for a
tightly verify the model transformation. Novel passes can easily be enabled for
verification by deriving from the `Verify` base and optionally tagging them with
the appropriate verification decorator.

```python
from onnx_passes.rewrite import RewriteRule
from onnx_passes.verify import Verify


class BreakAdds_v1(RewriteRule, Verify):
    """Break Add operations with distinct inputs on purpose."""

    @staticmethod
    def pattern(op, x, y):
        return op.Add(x, y)

    @staticmethod
    def check(context, x, y):
        return x != y

    @staticmethod
    def rewrite(op, x, y):
        return op.Add(x, x)
```
Applying this demo transformation with the command-line flow will produce
outputs roughly as follows (with the option `--browse` the failed model is shown
via [Netron](https://github.com/lutzroeder/netron)):
```
$ onnx-passes --browse model.onnx ... BreakAdds ... --verify.inputs ...
...
VerificationError: Output 'global_output__0' not as expected after 'BreakAdd_v1'
Serving 'debug-checkpoint.onnx' at http://localhost:20236
```

Currently three verification methods are implemented: equality-, tolerance- and
metric-based verification. Metric-based verification could become relevant once
introducing approximating optimizations which no longer guarantee model equality
and tracking degradation of a task-specific metric, such as accuracy, is more
indicative.

## Compiling Multi-Threshold Operators
The [`multithreshold`](onnx_passes/passes/multithreshold/__init__.py#L62)
submodule implements compilation of quantized activation functions to custom
[`MultiThreshold`](onnx_passes/ops/__init__.py#L523) operators for efficient
implementation with FINN as described in our corresponding publication:
```bibtex
@article{berganski2026compiling,
  title={Compiling Multi-Threshold Operators for Quantized Non-Monotonic and Fine-Granular Activations in Custom-Tailored DNN Accelerators},
  author={Berganski, Christoph and Jentzsch, Felix and Platzner, Marco},
  journal={arXiv preprint arXiv:2610.21825},
  year={2026}
}
```
