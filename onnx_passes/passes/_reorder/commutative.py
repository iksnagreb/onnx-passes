from onnx_passes.passes._base import RewriteRuleSetTemplate, Transformation
from onnx_passes.passes._verify import Verify

import onnx_ir as ir

from onnxscript import GraphBuilder
from onnxscript.rewriter.pattern import Pattern


def match_constant(_, value: ir.Value) -> bool:
    """Value level checker for constant values."""
    return ir.convenience.get_const_tensor(value) is not None


class ReorderCommutative_v1(RewriteRuleSetTemplate, Verify):
    """Reorder commutative operations to move constants to the right."""

    patterns = (
        lambda op: op.Add,
        lambda op: op.Mul,
        lambda op: op.Max,
        lambda op: op.Min,
        lambda op: op.Or,
        lambda op: op.And,
        lambda op: op.Xor,
        lambda op: op.BitwiseOr,
        lambda op: op.BitwiseAnd,
        lambda op: op.BitwiseXor,
        lambda op: op.Equal,
    )

    @staticmethod
    def pattern(partial, op, y):
        return partial(op)(match_constant, y, _outputs=["out"])

    @staticmethod
    def check(context, y, out):
        return not match_constant(context, y)

    @staticmethod
    def rewrite(partial, op, y, out):
        return partial(op)(y, out.producer().inputs[0])


class ReorderWideCommutative_v1(Transformation, Verify):
    """Reorder wide commutative operations to group reoccurring inputs.

    Note: This also assumes associativity as a wide multi-input operator pattern
    is matched as arbitrary groupings of binary operations.
    """

    patterns = (
        lambda op: op.Add,
        lambda op: op.Mul,
        lambda op: op.Max,
        lambda op: op.Min,
        lambda op: op.Or,
        lambda op: op.And,
        lambda op: op.Xor,
        lambda op: op.BitwiseOr,
        lambda op: op.BitwiseAnd,
        lambda op: op.BitwiseXor,
    )

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Graph builder and derived OpBuilder for inserting replacement ops into
        # the existing graph
        builder = GraphBuilder(model.graph)
        op = builder.op

        modified = False

        # Check each node for being one of the patterns and try to expand a tree
        # of likewise operations to reorder the inputs.
        for node in ir.traversal.RecursiveGraphIterator(
                model.graph, reverse=True
        ):
            for pattern in self.patterns:
                # Local pattern matching without ensuring removability of the
                # matched nodes (not using this with a rewrite rule)
                rule = Pattern(
                    lambda op, x: pattern(op)(  # noqa: op, x
                        x, _allow_other_inputs=True
                    )
                )

                def match(n: ir.Node) -> bool:
                    return bool(
                        rule.match(
                            model, model.graph, n,
                            check_nodes_are_removable=False
                        )
                    )

                # If the node matched the pattern, try to expand the tree and
                # reorder
                if match(node):
                    # Keep track of all inputs to the pattern grouped by their
                    # respective IR Value
                    inputs: dict[ir.Value, list[ir.Value]] = {}

                    # Breadth-first search for all inputs to the wide operator
                    # rooted at the node
                    nodes = [node]

                    while nodes:
                        for x in nodes.pop(0).inputs:
                            if x is not None:
                                if (producer := x.producer()) is None:
                                    inputs.setdefault(x, []).append(x)
                                elif match(producer):
                                    nodes.append(producer)
                                else:
                                    inputs.setdefault(x, []).append(x)

                    # Sort all inputs by number of occurrences to have the most
                    # frequent input at the end
                    sorted_inputs = dict(
                        sorted(inputs.items(), key=lambda item: len(item[1]))
                    )

                    if list(sorted_inputs) == list(inputs):
                        continue

                    # Flatten the sorted input list and reject reordering of
                    # binary or even unary operations
                    inputs: list[ir.Value] = [
                        x for values in sorted_inputs.values() for x in values
                    ]

                    if len(inputs) <= 2:
                        continue

                    inputs = list(reversed(inputs))

                    # Insert the replacement pattern as a wide chain of the
                    # operator with inputs reordered according to multiplicity
                    x = inputs[0]

                    for value in inputs[1:]:
                        x = pattern(op)(value, x)

                    ir.convenience.replace_all_uses_with(
                        node.outputs[0],
                        x,
                        replace_graph_outputs=True
                    )

                    modified = True

        return ir.passes.PassResult(model, modified)
