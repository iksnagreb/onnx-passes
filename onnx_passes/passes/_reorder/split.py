from onnx_passes.passes._base import Transformation, Sequential
from onnx_passes.passes._verify import Verify

from onnx_passes.traits.elementwise import produced_by_elementwise

import onnx_ir as ir

from onnxscript import GraphBuilder
from onnxscript.rewriter import pattern


def elementwise_split_pattern(op):
    """Pattern matching elementwise-split combination with optional inputs."""
    return op.Split(produced_by_elementwise, _allow_other_inputs=True)


class MoveElementwisePastSplit_v1(Transformation, Verify):
    """Reorder elementwise operations to follow splitting where applicable."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Graph builder and derived OpBuilder for inserting replacement ops into
        # the existing graph
        builder = GraphBuilder(model.graph)
        op = builder.op

        modified = False

        # Local pattern matching of elementwise-split combinations without
        # ensuring removability of the matched nodes (not using this with a
        # rewrite rule)
        rule = pattern.Pattern(elementwise_split_pattern)

        def match_elementwise_split(n: ir.Node) -> bool:
            return bool(
                rule.match(
                    model, model.graph, n, check_nodes_are_removable=False
                )
            )

        # Apply the transformation to matching elementwise-split patterns with
        # static second input (if present)
        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            if match_elementwise_split(node):
                x, *split = node.inputs

                if split:
                    if split[0].shape is None or split[0].shape.is_dynamic():
                        continue

                # Split each input to the elementwise operation after expanding
                # to the full output shape
                inputs = []

                for value in (elementwise := x.producer()).inputs:  # noqa
                    inputs.append(
                        op.Split(
                            op.Expand(
                                value, op.Shape(x)
                            ),
                            *split,
                            **node.attributes,
                            # Number of outputs must be specified to the graph
                            # builder, otherwise the splits are not iterable
                            _outputs=len(node.outputs)
                        )
                    )

                # Insert a new elementwise operator for each of the splits and
                # rewire the graph to consume the replacement
                for xs, out in zip(zip(*inputs), node.outputs):
                    ir.convenience.replace_all_uses_with(
                        out,
                        getattr(op, elementwise.op_type)(  # noqa: not None
                            *xs, **elementwise.attributes  # noqa: not None
                        ),
                        replace_graph_outputs=True
                    )

                modified = True

        return ir.passes.PassResult(model, modified)


class ReorderSplitLoop_v1(Sequential, Transformation):
    """Exhaustively apply split reordering transformations."""

    passes = [
        MoveElementwisePastSplit_v1
    ]

    exhaustive = True
