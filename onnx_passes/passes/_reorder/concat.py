from onnx_passes.passes._base import Transformation, RewriteRule, Sequential
from onnx_passes.passes._verify import Verify

from onnx_passes.traits.elementwise import produced_by_elementwise

import onnx_ir as ir

from onnxscript import GraphBuilder
from onnxscript.rewriter import pattern

from abc import ABC
from typing import Callable, Any


class ConcatMatchingIdentity(RewriteRule, ABC):
    """Template: Concatenates matching auxiliary identity operations.

    This is an auxiliary rewrite rule to make MoveElementwisePastConcat_v1 apply
    more often. If this does not allow for more reordering, the identities will
    be removed by the next round of identity elimination passes.
    """

    operator: Callable
    identity: Any

    @staticmethod
    def pattern(op):
        return op.Concat(_allow_other_inputs=True, _outputs=["out"])

    _matched: ir.Value | None

    def check(self, op, out):
        # Construct some adhoc pattern to match the operator within the current
        # match context, i.e, we already know this is a Concat but need to match
        # the secondary template pattern at each input.
        rule = pattern.Pattern(self.operator)

        def match_operator(n: ir.Node) -> bool:
            return bool(
                rule.match(
                    op.model, op.model.graph, n, check_nodes_are_removable=False
                )
            )

        # Match on the first input produced by the operator pattern and check
        # whether there is at least one other input not produced by the pattern.
        self._matched = None

        for value in out.producer().inputs:
            if ir.convenience.get_const_tensor(value) is None:
                if match_operator(value.producer()):
                    self._matched = value

        if self._matched is not None:
            for other in out.producer().inputs:
                if other.producer() is None:
                    return True

                if not match_operator(other.producer()):
                    return True

        return False

    def rewrite(self, op, out):
        # For all input values which are not the match value produced by the
        # elementwise operation, insert the matching identity
        for i, value in enumerate(inputs := list(out.producer().inputs)):
            if value != self._matched:
                inputs[i] = self.operator(
                    op,
                    value,
                    # Ensure matching input type
                    op.CastLike(
                        op.Constant(value=ir.tensor(self.identity)),
                        value
                    )
                )

        return op.Concat(*inputs, **out.producer().attributes)


class ConcatMatchingAddIdentity_v1(ConcatMatchingIdentity, Verify):
    """Insert addition for concat with Add only on some branches."""

    identity = 0

    @staticmethod
    def operator(op, x, y):
        return op.Add(x, y)

    @property
    def commute(self) -> bool:
        return True


class ConcatMatchingMulIdentity_v1(ConcatMatchingIdentity, Verify):
    """Insert multiplication for concat with Mul only on some branches."""

    identity = 1

    @staticmethod
    def operator(op, x, y):
        return op.Mul(x, y)

    @property
    def commute(self) -> bool:
        return True


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
                    replace_graph_outputs=True
                )

                modified = True

        return ir.passes.PassResult(model, modified)


class ReorderConcatLoop_v1(Sequential, Transformation):
    """Exhaustively apply concat reordering transformations."""

    passes = [
        ConcatMatchingAddIdentity_v1,
        ConcatMatchingMulIdentity_v1,
        MoveElementwisePastConcat_v1
    ]

    exhaustive = True