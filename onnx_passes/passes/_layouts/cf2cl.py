from onnx_passes.passes.base import Transformation, RewriteRule
from onnx_passes.passes.verify import Verify, tolerance

from onnx_passes.passes._layouts import LayoutError, layout_converter_pattern

from onnx_passes.ops import DOMAIN as CUSTOM_DOMAIN

from onnxscript import GraphBuilder
from onnxscript.rewriter import pattern

import onnx_ir as ir
import numpy as np


class GraphInputsToChannelsLast_v1(Transformation, Verify):
    """Insert channels-first to channels-last converter for graph inputs.

    This applies to all graph inputs of rank three or more, tensors of rank 0, 1
    or 2 have layouts (), N, NxC, i.e., channels-first/-last are the same.
    """

    def requires(self, model: ir.Model):
        """Requires that no input is annotated with an incompatible layout."""

        rule = pattern.Pattern(layout_converter_pattern)

        def match_layout_converter(n: ir.Node) -> bool:
            return bool(
                rule.match(
                    model, model.graph, n, check_nodes_are_removable=False
                )
            )

        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            if match_layout_converter(node):
                if (x := node.inputs[0]) in model.graph.inputs:
                    if (assumes := node.attributes.get("assumes")) is not None:
                        if (assumes := assumes.as_strings())[1] != "C":
                            raise LayoutError(
                                f"Cannot change to channels-last:"
                                f" Input {x.name} assumes layout {assumes}"
                            )

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Graph builder and derived OpBuilder for inserting replacement ops into
        # the existing graph
        builder = GraphBuilder(model.graph)
        op = builder.op

        modified = False

        # Consider all inputs of rank three or more, i.e., all with meaningful
        # distinction between channel and non-channel
        for inp in model.graph.inputs:
            if (rank := len(inp.shape)) >= 3:
                # Insert data layout conversion/annotation: This is effectively
                # a pair of transposes together acting as the identity
                x = op.Transpose(
                    converter := op.LayoutConverter(
                        inp,
                        assumes=[
                            "N", "C", *(f"D{n}" for n in range(1, rank - 1))
                        ],
                        perm=[
                            0, *range(2, rank), 1
                        ],
                        _domain=CUSTOM_DOMAIN
                    ),
                    perm=[
                        0, rank - 1, *range(1, rank - 1)
                    ]
                )

                # Each consumer of the original input must be rewired to consume
                # the new converter output
                for consumer in inp.consumers():
                    # Do not construct a cycle by replacing the converter input
                    # with the converter output...
                    if consumer != converter.producer():
                        # Might actually be consumed multiple times...
                        for index, value in enumerate(consumer.inputs):
                            if value == inp:
                                consumer.replace_input_with(index, x)

                if self.config.logging.verbose:
                    converter = converter.producer()

                    layout = converter.attributes.get("assumes").as_strings()
                    perm = converter.attributes.get("perm").as_ints()

                    print(
                        f"Inserting layout conversion for {inp.name}:"
                        f" {layout} to {tuple(layout[i] for i in perm)}"
                    )

                modified = True

        return ir.passes.PassResult(model, modified)


class GraphOutputsFromChannelsLast_v1(Transformation, Verify):
    """Insert channels-first to channels-last converter for graph outputs.

    This applies to all graph outputs of rank three or more, tensors of rank 0,
    1 or 2 have layouts (), N, NxC, i.e., channels-first/-last are the same.
    """

    def requires(self, model: ir.Model):
        """Requires that no output is annotated with an incompatible layout."""

        rule = pattern.Pattern(layout_converter_pattern)

        def match_layout_converter(n: ir.Node) -> bool:
            return bool(
                rule.match(
                    model, model.graph, n, check_nodes_are_removable=False
                )
            )

        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            if match_layout_converter(node):
                if (x := node.outputs[0]) in model.graph.outputs:
                    if (assumes := node.attributes.get("assumes")) is not None:
                        if (assumes := assumes.as_strings())[-1] != "C":
                            if ((perm := node.attributes.get("perm"))
                                    is not None):
                                assumes = [assumes[i] for i in perm.as_ints()]

                            raise LayoutError(
                                f"Cannot change to channels-last:"
                                f" Output {x.name} permutes layout to {assumes}"
                            )

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Graph builder and derived OpBuilder for inserting replacement ops into
        # the existing graph
        builder = GraphBuilder(model.graph)
        op = builder.op

        modified = False

        # Consider all outputs of rank three or more, i.e., all with meaningful
        # distinction between channel and non-channel
        for out in model.graph.outputs:
            if (rank := len(out.shape)) >= 3:
                # Insert data layout conversion/annotation: This is effectively
                # a pair of transposes together acting as the identity
                x = op.LayoutConverter(
                    op.Transpose(
                        out,
                        perm=[
                            0, *range(2, rank), 1
                        ]
                    ),
                    assumes=[
                        "N", *(f"D{n}" for n in range(1, rank - 1)), "C"
                    ],
                    perm=[
                        0, rank - 1, *range(1, rank - 1)
                    ],
                    _domain=CUSTOM_DOMAIN
                )

                # Rewire the graph to use the converted output instead of the
                # original output
                model.graph.outputs[model.graph.outputs.index(out)] = x

                if self.config.logging.verbose:
                    converter = x.producer()

                    layout = converter.attributes.get("assumes").as_strings()
                    perm = converter.attributes.get("perm").as_ints()

                    print(
                        f"Inserting layout conversion for {out.name}:"
                        f" {layout} to {tuple(layout[i] for i in perm)}"
                    )

                modified = True

        return ir.passes.PassResult(model, modified)


@tolerance
class FlattenMatMulToChannelsLast_v1(RewriteRule, Verify):
    """Switch layout of Flatten-MatMul from channels-first to channels-last.

    This implicitly assumes Flatten-MatMul to be in channels-first layout before
    applying this pass. Applying this to a graph already in any other layout can
    result in layouts that are neither channels-last nor -first.

    This applies to Flatten-MatMul with inputs of rank three or more, tensors of
    rank 0, 1 or 2 have layouts (), N, NxC.
    """

    @staticmethod
    def pattern(op, x, weights, shape):
        return op.MatMul(
            op.Reshape(
                x,
                shape
            ),
            weights
        )

    @staticmethod
    def check(context, x, weights, shape):
        if x.shape is None or x.shape.is_dynamic() or len(x.shape) <= 2:
            return False

        if (shape := ir.convenience.get_const_tensor(shape)) is not None:
            if tuple(shape.numpy()) == (x.shape[0], np.prod(x.shape[1:])):
                return ir.convenience.get_const_tensor(weights) is not None

        return False

    @staticmethod
    def rewrite(op, x, weights, shape):
        return op.MatMul(
            op.Reshape(
                op.Transpose(
                    op.Reshape(
                        x,
                        op.Constant(
                            value_ints=[
                                *x.shape[0:2],
                                int(np.prod(x.shape[2:]))
                            ]
                        )
                    ),
                    perm=[0, 2, 1]
                ),
                shape
            ),
            op.Reshape(
                op.Transpose(
                    op.Reshape(
                        weights,
                        op.Constant(
                            value_ints=[
                                # Note: keep leading batch dimensions of the
                                # weights for batch matrix-matrix multiplication
                                int(np.prod(weights.shape[:-2])),
                                x.shape[1],
                                -1,
                                weights.shape[-1]
                            ]
                        )
                    ),
                    perm=[0, 2, 1, 3]
                ),
                op.Shape(weights)
            )
        )
