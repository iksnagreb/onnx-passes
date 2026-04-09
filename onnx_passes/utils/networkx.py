# NetworkX graph structures and algorithms
import networkx as nx
# ONNX IR model wrapper, node iterator
import onnx_ir as ir


def onnx_to_nx(model: ir.Model) -> nx.DiGraph:
    """Constructs a NetworkX directed graph from the ONNX model."""

    # Start with an empty graph
    graph = nx.DiGraph()

    # Insert all global inputs and outputs as nodes into the NetworkX graph
    for value in [*model.graph.inputs, *model.graph.outputs]:
        graph.add_node(value)

    # Insert all nodes from the ONNX model as nodes into the NetworkX graph
    for node in ir.traversal.RecursiveGraphIterator(model.graph):
        graph.add_node(node)

    # Connect consumer and producer nodes
    for producer in graph.nodes:
        for consumer in graph.nodes:
            # Connect consumer Nodes to the producer nodes of the inputs
            if isinstance(consumer, ir.Node):
                for value in consumer.inputs:
                    # Connect producer nodes to the consumer of the outputs
                    if isinstance(producer, ir.Node):
                        if value in producer.outputs:
                            graph.add_edge(producer, consumer, value=value)
                    # Connect producer values directly to the consumer
                    if isinstance(producer, ir.Value):
                        if value == producer:
                            graph.add_edge(producer, consumer, value=value)
            # Connect consumer values directly to the producer
            if isinstance(value := consumer, ir.Value):
                # Connect producer nodes to the consumer of the outputs
                if isinstance(producer, ir.Node):
                    if value in producer.outputs:
                        graph.add_edge(producer, consumer, value=value)

    # NetworkX graph representation of the ONNX model graph
    return graph


def is_isomorphic(model: ir.Model, other: ir.Model) -> bool:
    """Test whether two model graphs are isomorphic to each other."""
    return nx.is_isomorphic(onnx_to_nx(model), onnx_to_nx(other))
