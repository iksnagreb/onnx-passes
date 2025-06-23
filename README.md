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

## Installation
```
pip install onnx-passes
```

## Tightly Verified Transformations
When successively transforming and optimizing a model, it is important to ensure
that the resulting model still matches the original model (at least within some
reasonable tolerance). To do this, the FINN framework injects verification steps
at a coarse schedule into the flow by evaluating the model and comparing the
outputs to a known reference. This project experiments with taking this a step
further by allowing to tightly verify the model after each individual pass
application. As a pass developer, novel passes can easily be enabled for
verification by tagging them with the appropriate verification decorator.
```python
# This transformation rewires any addition with two distinct inputs to add the
# first input to itself while ignoring the second one. This is not a valid
# equivalent transformation. However, as it is tagged to verify equality,
# applying this to any model containing Adds will raise a VerificationError
@passes.verify.equality
@passes.register()
class BreakAdds(Transformation, RewriteRulePass):
    def pattern(self, op, x, y):
        return op.Add(x, y)

    def check(self, _, x, y):
        return x != y

    def rewrite(self, op, x, y):
        return op.Add(x, x)
```
Applying this demo transformation with the command-line flow will produce
outputs roughly as follows:
```
$ onnx-passes -c cfg.yaml -o out.onnx model.onnx ... BreakAdds ...
...
Applied 1 of general pattern rewrite rules.
PostconditionError: Post-condition for pass 'BreakAdds' failed
VerificationError: global_output_0 not as expected
```

Currently three verification methods are implemented: equality-, tolerance- and
metric-based verification, where equality- and tolerance-based roughly
correspond to what is implemented in FINN, where the novel metric-based method
allows the user to inject metrics and accepted metric ranges for verification.
This method could become relevant once introducing approximating optimizations
which no longer guarantee model equality and tracking degradation of a
task-specific metric, such as accuracy, is more indicative.

## Roadmap and Ideas
- [x] Skeleton of pass base-classes: *Analysis*, *Annotation*, *Transformation*,
*RewriteRulePass* and *ComposePass*
- [x] Basic cleanup and constant-folding transformations based on ONNX IR and
ONNX Script
- [x] Infrastructure for tight verification of transformations via
[ONNX Runtime](https://github.com/microsoft/onnxruntime) model execution
- [x] Simple command-line script `onnx-passes` for experimenting with
pass-application and verification
- [ ] QONNX interoperability: `Quant`, `BipolarQuant`, `Trunc` and
`MultiThreshold` operators, arbitrary-precision integer data types _(work in
progress)_ 
- [ ] Reimplement streamlining similar to FINN _(work in progress)_ 
- [ ] Annotating and resolving pass dependencies and follow-up post-processing 
transformations, e.g., _streamline_ **needs** _shape-inference_ and **must be 
followed by** _cleanup_
- [ ] Verification of structural properties of a graph, e.g., cycle-free, 
isomorphic to..., connected, ...
- [ ] ...
