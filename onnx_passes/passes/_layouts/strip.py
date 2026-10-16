from onnx_passes.passes.base import Transformation
from onnx_passes.passes._layouts import layout_converter_pattern

from onnxscript import GraphBuilder
from onnxscript.rewriter import pattern

import onnx_ir as ir


class StripLayoutConverter_v1(Transformation):
    """Strip input/output layout conversion from the graph.

    This pass changes the model input/output shape and thus cannot be verified
    as input/output necessarily will not match. Pass sequences containing this
    can also not be verified.

    This pass should be applied standalone, as part of an unverified sequence,
    or in a staggered flow, where reference inputs/outputs are converted after
    applying this pass - all passes before/after could then be verified.
    """

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Graph builder and derived OpBuilder for inserting replacement ops into
        # the existing graph
        builder = GraphBuilder(model.graph)
        op = builder.op

        modified = False

        rule = pattern.Pattern(layout_converter_pattern)

        def match_layout_converter(n: ir.Node) -> bool:
            return bool(
                rule.match(
                    model, model.graph, n, check_nodes_are_removable=False
                )
            )

        # First strip any input/output layout converter. This actually changes
        # the model interface.
        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            if match_layout_converter(node):
                # On the input side (and if this is not also an output), we need
                # to insert a new value and rewire consumers to use this.
                if (x := node.inputs[0]) in model.graph.inputs:
                    if (y := node.outputs[0]) not in model.graph.outputs:
                        ir.convenience.replace_all_uses_with(
                            y,
                            new := ir.Value(
                                type=y.type, shape=y.shape, name=y.name
                            )
                        )
                        model.graph.inputs[model.graph.inputs.index(x)] = new

                        if self.config.logging.verbose:
                            layout = node.attributes.get("assumes").as_strings()
                            perm = node.attributes.get("perm").as_ints()

                            print(
                                f"Stripping layout conversion from {x.name}:"
                                f" {layout} to {tuple(layout[i] for i in perm)}"
                            )

                        modified = True

                # On the output side (also if this is an input), we can simply
                # skip over the layout conversion.
                if (y := node.outputs[0]) in model.graph.outputs:
                    ir.convenience.replace_all_uses_with(
                        y, node.inputs[0], replace_graph_outputs=True
                    )

                    if self.config.logging.verbose:
                        layout = node.attributes.get("assumes").as_strings()
                        perm = node.attributes.get("perm").as_ints()

                        print(
                            f"Stripping layout conversion from {x.name}:"
                            f" {layout} to {tuple(layout[i] for i in perm)}"
                        )

                    modified = True

        # Turn all remaining, internal layout converters into standard Transpose
        # operations - no custom LayoutConverter should remain in the graph.
        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            if match_layout_converter(node):
                ir.convenience.replace_all_uses_with(
                    node.outputs[0],
                    op.Transpose(
                        node.inputs[0],
                        perm=node.attributes.get("perm")
                    )
                )

                modified = True

        return ir.passes.PassResult(model, modified)
