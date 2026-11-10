from pydantic import BaseModel, computed_field

import onnx_ir as ir
import numpy as np


def count_mac_matmul(node: ir.Node) -> int | None:
    """Count MAC operations of MatMul if shapes are statically known."""

    if len(node.inputs) < 2 or (weights := node.inputs[1]) is None:
        return None

    if (weights_shape := weights.shape) is None or weights_shape.is_dynamic():
        return None

    if (out_shape := node.outputs[0].shape) is None or out_shape.is_dynamic():
        return None

    # Per element of the output, there are MAC operations along the innermost
    # dimension of the left hand side input
    return int(np.prod(out_shape.numpy()) * np.prod(weights_shape.numpy()[-1]))


def count_mac_gemm(node: ir.Node) -> int | None:
    """Count MAC operations of Gemm if shapes are statically known."""

    if len(node.inputs) < 2 or (x := node.inputs[0]) is None:
        return None

    if (x_shape := x.shape) is None or x_shape.is_dynamic():
        return None

    if (out_shape := node.outputs[0].shape) is None or out_shape.is_dynamic():
        return None

    # Per element of the output, there are MAC operations along the innermost
    # dimension of the left hand side input
    if (trans := node.attributes.get("transA")) is None or trans.as_int() == 0:
        return int(np.prod(out_shape.numpy()) * np.prod(x_shape.numpy()[-1]))

    return int(np.prod(out_shape.numpy()) * np.prod(x_shape.numpy()[-2]))


def count_mac_conv(node: ir.Node) -> int | None:
    """Count MAC operations of Conv if shapes are statically known."""

    if len(node.inputs) < 2 or (weights := node.inputs[1]) is None:
        return None

    if (weights_shape := weights.shape) is None or weights_shape.is_dynamic():
        return None

    if (out_shape := node.outputs[0].shape) is None or out_shape.is_dynamic():
        return None

    return int(np.prod(out_shape.numpy()) * np.prod(weights_shape.numpy()[1:]))


def count_mac_by_op_type(node: ir.Node) -> int | None:
    """Count MAC operations of the node according to its op-type."""

    if node.op_type == "Conv":
        return count_mac_conv(node)

    if node.op_type == "MatMul":
        return count_mac_matmul(node)

    if node.op_type == "Gemm":
        return count_mac_gemm(node)

    return None


# Shortcuts for annotating the different count breakdown categories: These are
# all combinations of dictionaries and tuples mapping some identifier to integer
OP_TYPE = tuple[str, str, str]
COUNT_PER_NODE = dict[str | None, int]
COUNT_PER_DTYPE = dict[ir.DataType | None, int]
COUNT_PER_OP_TYPE = dict[OP_TYPE | None, int]
COUNT_PER_ATTR_TYPE = dict[ir.AttributeType | None, int]


class MACCounts(BaseModel, use_attribute_docstrings=True):
    """Model MAC operation counts broken down into categories."""

    per_node: COUNT_PER_NODE = {}
    """Number of MAC operations per node."""

    per_op_type: COUNT_PER_OP_TYPE = {}
    """Number of MAC operations per operator type."""

    @computed_field
    @property
    def total(self) -> int:
        """Total number of MAC operations in the model."""
        return sum(self.per_op_type.values())

    @classmethod
    def from_model(cls, model: ir.Model) -> "MACCounts":
        """Analyze MAC operation counts from model."""

        counts = MACCounts()

        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            if (count := count_mac_by_op_type(node)) is not None:
                counts.per_node[node.name] = count

                if node.op_identifier() not in counts.per_op_type:
                    counts.per_op_type[node.op_identifier()] = 0

                counts.per_op_type[node.op_identifier()] += count

        return counts


class OperationCounts(BaseModel, use_attribute_docstrings=True):
    """Model operation counts broken down into categories."""

    per_op_type: COUNT_PER_OP_TYPE = {}
    """Number of operators per operator type."""

    mac_ops: MACCounts = MACCounts()
    """Breakdown of number of MAC operations"""

    @computed_field
    @property
    def total(self) -> int:
        """Total number of operators, i.e., ir.Node, in the model."""
        return sum(self.per_op_type.values())

    @classmethod
    def from_model(cls, model: ir.Model) -> "OperationCounts":
        """Analyze operation counts from model."""

        counts = OperationCounts()

        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            if node.op_identifier() not in counts.per_op_type:
                counts.per_op_type[node.op_identifier()] = 0

            counts.per_op_type[node.op_identifier()] += 1

        counts.mac_ops = MACCounts.from_model(model)

        return counts


class EffectiveIntegerCounts(BaseModel, use_attribute_docstrings=True):
    """Effectively integer-valued model parameter counts by categories."""

    per_type: COUNT_PER_DTYPE = {}
    """Number of effectively integer-valued parameters per data type."""

    per_node: dict[str | None, COUNT_PER_DTYPE] = {}
    """Number of effectively integer-valued parameters per node."""

    per_op_type: dict[OP_TYPE, COUNT_PER_DTYPE] = {}
    """Number of effectively integer-valued parameters per operator type."""

    @computed_field
    @property
    def total(self) -> int:
        """Total number of effectively integer-valued parameters."""
        return sum(self.per_type.values())

    @classmethod
    def from_model(cls, model: ir.Model) -> "EffectiveIntegerCounts":
        """Analyze effectively integer-valued parameter counts from model."""

        counts = EffectiveIntegerCounts()

        for value in ir.convenience.create_value_mapping(model.graph).values():
            if (param := ir.convenience.get_const_tensor(value)) is not None:
                try:
                    # Effectively integer-valued: Rounding makes no difference
                    if np.all(np.round(param.numpy()) == param.numpy()):
                        count = param.numpy().size

                        # Per data type parameter counts
                        if value.dtype not in counts.per_type:
                            counts.per_type[value.dtype] = 0

                        counts.per_type[value.dtype] += count

                        for c in value.consumers():
                            # Per data type per node parameter counts
                            if c.name not in counts.per_node:
                                counts.per_node[c.name] = {}

                            per_type = counts.per_node[c.name]

                            if value.dtype not in per_type:
                                per_type[value.dtype] = 0

                            per_type[value.dtype] += count

                            # Per data type per operator type parameter counts
                            if c.op_identifier() not in counts.per_op_type:
                                counts.per_op_type[c.op_identifier()] = {}

                            per_type = counts.per_op_type[c.op_identifier()]

                            if value.dtype not in per_type:
                                per_type[value.dtype] = 0

                            per_type[value.dtype] += count

                except:  # noqa: Broad on purpose to ignore non-numeric values
                    pass

        return counts


class NonzeroCounts(BaseModel, use_attribute_docstrings=True):
    """Non-zero model parameter counts by categories."""

    per_type: COUNT_PER_DTYPE = {}
    """Number of non-zero parameters per data type."""

    per_node: dict[str | None, COUNT_PER_DTYPE] = {}
    """Number of non-zero parameters per node."""

    per_op_type: dict[OP_TYPE | None, COUNT_PER_DTYPE] = {}
    """Number of non-zero parameters per operator type."""

    @computed_field
    @property
    def total(self) -> int:
        """Total number of non-zero parameters."""
        return sum(self.per_type.values())

    @classmethod
    def from_model(cls, model: ir.Model) -> "NonzeroCounts":
        """Analyze non-zero parameter counts from model."""

        counts = NonzeroCounts()

        for value in ir.convenience.create_value_mapping(model.graph).values():
            if (param := ir.convenience.get_const_tensor(value)) is not None:
                count = int(np.sum(param.numpy() != 0))

                # Per data type parameter counts
                if value.dtype not in counts.per_type:
                    counts.per_type[value.dtype] = 0

                counts.per_type[value.dtype] += count

                for c in value.consumers():
                    # Per data type per node parameter counts
                    if c.name not in counts.per_node:
                        counts.per_node[c.name] = {}

                    per_type = counts.per_node[c.name]

                    if value.dtype not in per_type:
                        per_type[value.dtype] = 0

                    per_type[value.dtype] += count

                    # Per data type per operator type parameter counts
                    if c.op_identifier() not in counts.per_op_type:
                        counts.per_op_type[c.op_identifier()] = {}

                    per_type = counts.per_op_type[c.op_identifier()]

                    if value.dtype not in per_type:
                        per_type[value.dtype] = 0

                    per_type[value.dtype] += count

        return counts


class ParameterCounts(BaseModel, use_attribute_docstrings=True):
    """Model parameter counts broken down into categories."""

    per_type: COUNT_PER_DTYPE = {}
    """Number of parameters per data type."""

    per_node: dict[str | None, COUNT_PER_DTYPE] = {}
    """Number of parameters per node."""

    per_op_type: dict[OP_TYPE | None, COUNT_PER_DTYPE] = {}
    """Number of parameters per operator type."""

    effective_integer: EffectiveIntegerCounts = EffectiveIntegerCounts()
    """Breakdown of effectively integer-valued parameters."""

    nonzero: NonzeroCounts = NonzeroCounts()
    """Breakdown of non-zero parameters."""

    @computed_field
    @property
    def total(self) -> int:
        """Total number of parameters, i.e., constants, in the model."""
        return sum(self.per_type.values())

    @classmethod
    def from_model(cls, model: ir.Model) -> "ParameterCounts":
        """Analyze parameter counts from model."""

        counts = ParameterCounts()

        for value in ir.convenience.create_value_mapping(model.graph).values():
            if (param := ir.convenience.get_const_tensor(value)) is not None:
                # Per data type parameter counts
                if value.dtype not in counts.per_type:
                    counts.per_type[value.dtype] = 0

                counts.per_type[value.dtype] += param.numpy().size

                for c in value.consumers():
                    # Per data type per node parameter counts
                    if c.name not in counts.per_node:
                        counts.per_node[c.name] = {}

                    per_type = counts.per_node[c.name]

                    if value.dtype not in per_type:
                        per_type[value.dtype] = 0

                    per_type[value.dtype] += param.numpy().size

                    # Per data type per operator type parameter counts
                    if c.op_identifier() not in counts.per_op_type:
                        counts.per_op_type[c.op_identifier()] = {}

                    per_type = counts.per_op_type[c.op_identifier()]

                    if value.dtype not in per_type:
                        per_type[value.dtype] = 0

                    per_type[value.dtype] += param.numpy().size

        counts.effective_integer = EffectiveIntegerCounts.from_model(model)
        counts.nonzero = NonzeroCounts.from_model(model)

        return counts


class AttributeCounts(BaseModel, use_attribute_docstrings=True):
    """Model node attribute counts broken down into categories."""

    per_type: COUNT_PER_ATTR_TYPE = {}
    """Number of attributes per data type."""

    per_name: dict[str | None, int] = {}
    """Number of attributes per attribute name."""

    per_node: dict[str | None, COUNT_PER_ATTR_TYPE] = {}
    """Number of attributes per node."""

    per_op_type: dict[OP_TYPE | None, COUNT_PER_ATTR_TYPE] = {}
    """Number of attributes per operator type."""

    @computed_field
    @property
    def total(self) -> int:
        """Total number of attributes in the model."""
        return sum(self.per_type.values())

    @classmethod
    def from_model(cls, model: ir.Model) -> "AttributeCounts":
        """Analyze attribute counts from model."""

        counts = AttributeCounts()

        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            for attribute in node.attributes.values():
                # Per data type attribute counts
                if attribute.type not in counts.per_type:
                    counts.per_type[attribute.type] = 0

                counts.per_type[attribute.type] += 1

                # Per attribute name attribute counts
                if attribute.name not in counts.per_name:
                    counts.per_name[attribute.name] = 0

                counts.per_name[attribute.name] += 1

                # Per data type per node attribute counts
                if node.name not in counts.per_node:
                    counts.per_node[node.name] = {}

                per_type = counts.per_node[node.name]

                if attribute.type not in per_type:
                    per_type[attribute.type] = 0

                per_type[attribute.type] += 1

                # Per data type per operator type attribute counts
                if node.op_identifier() not in counts.per_op_type:
                    counts.per_op_type[node.op_identifier()] = {}

                per_type = counts.per_op_type[node.op_identifier()]

                if attribute.type not in per_type:
                    per_type[attribute.type] = 0

                per_type[attribute.type] += 1

        return counts


class ModelTopology(BaseModel, use_attribute_docstrings=True):
    """Summary of model topology, i.e., connectivity, by categories."""

    inputs: int = 0
    """Number of graph inputs."""

    outputs: int = 0
    """Number of graph outputs."""

    fan_in_per_node: COUNT_PER_NODE = {}
    """Number of non-constant inputs per node."""

    fan_out_per_node: COUNT_PER_NODE = {}
    """Number of outputs/consumers per node."""

    @computed_field
    @property
    def max_fan_in(self) -> int:
        """Maximum per node fan-in in the model."""
        return max(self.fan_in_per_node.values())

    @computed_field
    @property
    def max_fan_out(self) -> int:
        """Maximum per node fan-out in the model."""
        return max(self.fan_out_per_node.values())

    @computed_field
    @property
    def branching(self) -> bool:
        """Indicate whether the model has branching topology.

        A model is considered branching if any of its operations, i.e., nodes of
        the graph, has more than one consumer, i.e., the maximum fan-out is >1.
        """
        return self.max_fan_out > 1

    @computed_field
    @property
    def converging(self) -> bool:
        """Indicate whether the model has converging topology.

        A model is considered branching if any of its operations, i.e., nodes of
        the graph, has more than one producer, i.e., the maximum fan-in is >1.
        """
        return self.max_fan_in > 1

    @classmethod
    def from_model(cls, model: ir.Model) -> "ModelTopology":
        """Analyze topology from model."""

        counts = ModelTopology()

        counts.inputs = len(model.graph.inputs)
        counts.outputs = len(model.graph.outputs)

        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            # Fan-in per node counts all inputs of the node, counting
            # repeated inputs multiple times
            if node.name not in counts.fan_in_per_node:
                counts.fan_in_per_node[node.name] = 0

            for x in node.inputs:
                if x is not None and ir.convenience.get_const_tensor(x) is None:
                    counts.fan_in_per_node[node.name] += 1

            # Fan-out per node counts all consumers for all outputs of the
            # node, counting repeated consumers multiple times
            if node.name not in counts.fan_out_per_node:
                counts.fan_out_per_node[node.name] = 0

            for output in node.outputs:
                for consumer in output.consumers():
                    for x in consumer.inputs:
                        if x == output:
                            counts.fan_out_per_node[node.name] += 1

                for x in model.graph.outputs:
                    if x == output:
                        counts.fan_out_per_node[node.name] += 1

        # Treat global inputs as nodes for tracking the maximum fan-out in the
        # model
        for value in model.graph.inputs:
            if value.name not in counts.fan_out_per_node:
                counts.fan_out_per_node[value.name] = 0

            for consumer in value.consumers():
                for x in consumer.inputs:
                    if x == value:
                        counts.fan_out_per_node[value.name] += 1

            # Also consider direct pass-through connections from global input to
            # global outputs
            for x in model.graph.outputs:
                if x == value:
                    counts.fan_out_per_node[value.name] += 1

        # Treat global outputs as nodes for tracking the maximum fan-in in the
        # model (outputs have by definition a fan-in of 1)
        for value in model.graph.outputs:
            if value.name not in counts.fan_in_per_node:
                counts.fan_in_per_node[value.name] = 1

        return counts


class Summary(BaseModel, use_attribute_docstrings=True):
    """Model summary: Count operations, parameters, attributes, etc."""

    operation_counts: OperationCounts = OperationCounts()
    """Model operation counts broken down into categories."""

    parameter_counts: ParameterCounts = ParameterCounts()
    """Model parameter counts broken down into categories."""

    attribute_counts: AttributeCounts = AttributeCounts()
    """Model node attribute counts broken down into categories."""

    model_topology: ModelTopology = ModelTopology()
    """Summary of model topology, i.e., connectivity, by categories."""

    @classmethod
    def from_model(cls, model: ir.Model) -> "Summary":
        """Generate summary from model."""

        summary = Summary()

        summary.operation_counts = OperationCounts.from_model(model)
        summary.parameter_counts = ParameterCounts.from_model(model)
        summary.attribute_counts = AttributeCounts.from_model(model)

        summary.model_topology = ModelTopology.from_model(model)

        return summary
