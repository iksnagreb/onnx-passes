# ir.Value, ir.convenience.get_const_tensor
import onnx_ir as ir

# Common passes from ONNX IR for cleaning up constants: Prefer all constants to
# be initializers, but do not track these as graph inputs
from onnx_ir.passes.common import RemoveInitializersFromInputsPass

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# All algebraic passes are transformations derived from pattern-based rewrite
# rules
from onnx_passes.passes.base import Transformation, RewriteRuleSetPass

# Domain used by custom operators implemented with this library
from onnx_passes.ops import DOMAIN as CUSTOM_DOMAIN
# LayoutConverter custom operator implemented with this library
from onnx_passes.ops.layouts import LayoutConverter  # noqa: Used via registry

# Exception type indicating failure while inserting layout assumptions and/or
# conversions - currently does not do add anything ontop the base Exception.
class LayoutError(Exception):
    ...


# Finds the inverse permutation of the transpose operation
def _inverse_perm(perm: list[int]):
    return [perm.index(i) for i in range(len(perm))]


# Inserts LayoutConverter operators at the graph inputs and outputs as
# configured by the configuration dictionary
#
# Note: As this contains a call to RemoveInitializersFromInputsPass, which
# enables different ONNX Runtime constant optimizations, this pass does not
# preserve equality, only equality within tolerance.
@passes.verify.tolerance
@passes.register("convert-layouts")
class ConvertLayouts(Transformation):
    # List of input layout annotations from the configuration dictionary
    @property
    def input_layouts(self):
        return self.config.setdefault("layouts", {}).setdefault("inp", [])

    # List of output layout annotations from the configuration dictionary
    @property
    def output_layouts(self):
        return self.config.setdefault("layouts", {}).setdefault("out", [])

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Track whether any node actually changed
        modified = False

        # Layout conversion should not operate on initializers as we do not
        # necessarily know the order and can do this via transpose streamlining.
        RemoveInitializersFromInputsPass()(model)

        # Tape recording nodes, values and attributes to be inserted into the
        # graph
        tape = ir.tape.Tape()

        # Insert configured layout conversion for each input tensor: Layout
        # conversion is optional, trailing inputs can just be omitted
        for inp, layout in zip(model.graph.inputs, self.input_layouts):
            # Skip explicitly missing layout conversion, i.e., None or null
            if layout is None:
                continue

            # Both layout attributes are strictly required...
            if "perm" not in layout or "assumes" not in layout:
                raise LayoutError(
                    f"Illegal layout annotation for input {inp.name}: {layout}"
                )

            # While not enforced, the layout assumption should at least be
            # consistent with the layout permutation
            if len(layout["assumes"]) != len(layout["perm"]):
                raise LayoutError(
                    f"Inconsistent layout assumption {layout['assumes']}"
                    f" for permutation {layout['perm']} of input {inp.name}"
                )

            # New output connecting the layout converter to the original graph
            # input (or rather its consumers)
            output = ir.Value(type=inp.type, shape=inp.shape)

            # Each consumer of the original input must be rewired to consume the
            # new converter output
            for consumer in inp.consumers():
                # Might actually be consumed multiple times...
                for index, x in enumerate(consumer.inputs):
                    if x == inp:
                        consumer.replace_input_with(index, output)

            # Select the inverse permutation of the layout to forward to the
            # transpose
            perm = {"perm": _inverse_perm(layout["perm"])}

            # Create and record the layout converter operators: This is
            # effectively a pair of transposes together acting as the identity
            x = tape.op("LayoutConverter", [inp], layout, domain=CUSTOM_DOMAIN)
            _ = tape.op("Transpose", [x], perm, output=output)

        # Insert configured layout conversion for each output tensor: Layout
        # conversion is optional, trailing outputs can just be omitted
        for out, layout in zip(model.graph.outputs, self.output_layouts):
            # Skip explicitly missing layout conversion, i.e., None or null
            if layout is None:
                continue

            # Both layout attributes are strictly required...
            if "perm" not in layout or "assumes" not in layout:
                raise LayoutError(
                    f"Illegal layout annotation for output {out.name}: {layout}"
                )

            # While not enforced, the layout assumption should at least be
            # consistent with the layout permutation
            if len(layout["assumes"]) != len(layout["perm"]):
                raise LayoutError(
                    f"Inconsistent layout assumption {layout['assumes']}"
                    f" for permutation {layout['perm']} of output {out.name}"
                )

            # New output connecting the layout converter to the original graph
            # output (or rather its consumers)
            output = ir.Value(type=out.type, shape=out.shape)

            # Select the inverse permutation of the layout to forward to the
            # transpose
            perm = {"perm": _inverse_perm(layout["perm"])}

            # Create and record the layout converter operators: This is
            # effectively a pair of transposes together acting as the identity
            x = tape.op("LayoutConverter", [out], layout, domain=CUSTOM_DOMAIN)
            _ = tape.op("Transpose", [x], perm, output=output)

            # Rewire the graph output to connect to the converter output
            model.graph.outputs[model.graph.outputs.index(out)] = output

        # Only modify the graph if converter nodes are recorded
        if tape.nodes:
            # Insert all collected converter operators into the graph and ensure
            # topological sorting
            model.graph.extend(tape.nodes)
            model.graph.sort()
            # Mark the model as modified
            modified = True

        # Potentially modified model and indicator whether the model actually
        # changed
        return ir.passes.PassResult(model, modified)


# Absorbs transpose operations surrounding a layout converter: This can be used
# as a sink for transpose operations which propagated to the start/end of the
# graph via streamlining. The layout assumption attribute is adjusted according
# to the absorbed permutation.
@passes.verify.equality
@passes.register("absorb-layouts")
class AbsorbTransposeIntoLayoutConverter(Transformation, RewriteRuleSetPass):
    def pattern(self):
        # Matches Transpose in front of LayoutConverter
        def _transpose_convert(op, x, perm1, perm2, assumes):
            # First the input is transposed by an explicit permutation
            transpose = op.Transpose(x, perm=perm1)
            # Effectively again a transpose with another permutation while
            # annotating a semantic layout assumption
            return op.LayoutConverter(
                transpose, perm=perm2, assumes=assumes, _domain=CUSTOM_DOMAIN
            )

        # Matches LayoutConverter in front of Transpose
        def _convert_transpose(op, x, perm1, perm2, assumes):
            # Effectively a transpose with explicit permutation while annotating
            # a semantic layout assumption
            convert = op.LayoutConverter(
                x, perm=perm1, assumes=assumes, _domain=CUSTOM_DOMAIN
            )
            # Transpose again with another permutation without additional layout
            # annotation (should be inferrable, but we don't really care here)
            return op.Transpose(convert, perm=perm2)

        # Set of the two target patterns to match
        return _transpose_convert, _convert_transpose

    def rewrite(self):
        # Replaces Transpose in front of LayoutConverter
        def _transpose_convert(op, x, perm1, perm2, assumes):
            # Fuse permutations by permuting the first permutation according to
            # the second
            perm = [perm1.as_ints()[i] for i in perm2.as_ints()]
            # Absorb first permutation into the layout assumption by permuting
            # the layout according to the inverse (propagates layout backwards)
            assumes = [
                assumes.as_strings()[i] for i in _inverse_perm(perm1.as_ints())
            ]
            # Replacement: Layout converter with fused permutation and
            # propagated layout assumption
            return op.LayoutConverter(
                x, perm=perm, assumes=assumes, _domain=CUSTOM_DOMAIN
            )

        # Replaces LayoutConverter in front of Transpose
        def _convert_transpose(op, x, perm1, perm2, assumes):
            # Fuse permutations by permuting the first permutation according to
            # the second
            perm = [perm1.as_ints()[i] for i in perm2.as_ints()]
            # In this case the input layout assumption does not change
            assumes = assumes.as_strings()
            # Replacement: Layout converter with fused permutation and
            # same layout assumption
            return op.LayoutConverter(
                x, perm=perm, assumes=assumes, _domain=CUSTOM_DOMAIN
            )

        # Set of two replacement patterns
        return _transpose_convert, _convert_transpose

# TODO: Consider implementing checks to enforce layout assumptions and/or
#  propagation of layout annotations through the graph as value metadata?
