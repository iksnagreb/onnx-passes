from onnx_passes.passes._base import Transformation
from onnx_passes.passes._verify import Verify

from onnx_passes.traits.elementwise import produced_by_elementwise

import onnx_ir as ir

from onnxscript import GraphBuilder
from onnxscript.rewriter import pattern


def elementwise_concat_pattern(op):
    """Pattern matching elementwise-concat combination with optional inputs."""
    return op.Concat(produced_by_elementwise, _allow_other_inputs=True)


class MoveElementwisePastConcat_v1(Transformation, Verify):
    """Reorder elementwise operations to follow concat where applicable."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Graph builder and derived OpBuilder for inserting replacement ops into
        # the existing graph
        builder = GraphBuilder(model.graph)
        op = builder.op

        modified = False

        # Local pattern matching of elementwise-concat combinations without
        # ensuring removability of the matched nodes (not using this with a
        # rewrite rule)
        rule = pattern.Pattern(elementwise_concat_pattern)

        def match_elementwise_concat(n: ir.Node) -> bool:
            return bool(
                rule.match(
                    model, model.graph, n, check_nodes_are_removable=False
                )
            )

        # Apply the transformation to matching elementwise-concat pattern
        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            if match_elementwise_concat(node):
                # The first matched elementwise operator determines the
                # elementwise operation to match for the whole pattern
                elementwise = node.inputs[0].producer()  # noqa: not None

                def is_same_elementwise_op(other: ir.Node | None) -> bool:
                    if other is not None:
                        return other.op_type == elementwise.op_type  # noqa
                    return False

                # Concatenate the corresponding inputs from each of the
                # elementwise operators after expanding to a compatible shape
                inputs = []

                for x in node.inputs:
                    if x is None or not is_same_elementwise_op(x.producer()):
                        break

                    inputs.append([])

                    for value in x.producer().inputs:  # noqa: not None
                        inputs[-1].append(
                            op.Expand(
                                value,
                                op.Shape(x)
                            )
                        )

                # Not all inputs are produced by the same type of elementwise
                # operation
                if len(inputs) != len(node.inputs):
                    continue

                inputs = [
                    op.Concat(*xs, **node.attributes) for xs in zip(*inputs)
                ]

                # Insert a new elementwise operator for the concatenated inputs
                # and rewire the graph to consume the replacement
                ir.convenience.replace_all_uses_with(
                    node.outputs[0],
                    getattr(op, elementwise.op_type)(  # noqa: not None
                        *inputs, **elementwise.attributes  # noqa: not None
                    ),
                    True  # Replace graph outputs as well
                )

                modified = True

        return ir.passes.PassResult(model, modified)
