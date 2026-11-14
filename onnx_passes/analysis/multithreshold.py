from onnx_passes.traits.multithreshold import is_multithreshold

from pydantic import BaseModel, computed_field

import onnx_ir as ir
import numpy as np


class MultiThresholdSteps(BaseModel, use_attribute_docstrings=True):
    """Steps of MultiThreshold operators in the model.

    The number of steps a MultiThreshold operator is the number of thresholds
    irrespective of their associated weights.
    """

    per_node: dict[str | None, int] = {}
    """Steps per MultiThreshold operator in the model."""

    @computed_field
    @property
    def maximum(self) -> int | None:
        """Maximum steps of all MultiThreshold in the model."""
        if self.per_node:
            return max(self.per_node.values())

        return None

    @computed_field
    @property
    def average(self) -> float | None:
        """Average steps of all MultiThreshold in the model."""
        if self.per_node:
            return sum(self.per_node.values()) / len(self.per_node)

        return None

    @classmethod
    def from_model(cls, model: ir.Model) -> "MultiThresholdSteps":
        """Collect statistics from model."""

        steps = MultiThresholdSteps()

        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            if is_multithreshold(node):
                if (thresholds := node.inputs[1]) is None:
                    continue

                if (shape := thresholds.shape) is None or shape.is_dynamic():
                    continue

                steps.per_node[node.name] = shape[-1]  # noqa: not symbolic

        return steps


class MultiThresholdDepth(BaseModel, use_attribute_docstrings=True):
    """Depth of MultiThreshold operators in the model.

    The depth of a MultiThreshold operator is log2(steps + 1), i.e., the depth
    of the binary search tree representation.
    """

    per_node: dict[str | None, float] = {}
    """Depth per MultiThreshold operator in the model."""

    @computed_field
    @property
    def maximum(self) -> float | None:
        """Maximum depth of all MultiThresholds in the model."""
        if self.per_node:
            return max(self.per_node.values())

        return None

    @computed_field
    @property
    def average(self) -> float | None:
        """Average depth of all MultiThresholds in the model."""
        if self.per_node:
            return sum(self.per_node.values()) / len(self.per_node)

        return None

    @classmethod
    def from_model(cls, model: ir.Model) -> "MultiThresholdDepth":
        """Collect statistics from model."""

        depth = MultiThresholdDepth()
        steps = MultiThresholdSteps.from_model(model)

        for node, steps in steps.per_node.items():
            depth.per_node[node] = float(np.log2(steps + 1))

        return depth


class MultiThresholdRange(BaseModel, use_attribute_docstrings=True):
    """Range of MultiThreshold operators in the model.

    The range of a MultiThreshold operator is the difference between the lowest
    and the highest output it can produce, i.e., the difference between maximum
    and minimum of the cumulative sum of the step weights.
    """

    per_node: dict[str | None, float] = {}
    """Range per MultiThreshold operator in the model."""

    @computed_field
    @property
    def maximum(self) -> float | None:
        """Maximum range of all MultiThresholds in the model."""
        if self.per_node:
            return max(self.per_node.values())

        return None

    @computed_field
    @property
    def average(self) -> float | None:
        """Average range of all MultiThresholds in the model."""
        if self.per_node:
            return sum(self.per_node.values()) / len(self.per_node)

        return None

    @classmethod
    def from_model(cls, model: ir.Model) -> "MultiThresholdRange":
        """Collect statistics from model."""

        multithreshold_range = MultiThresholdRange()

        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            if is_multithreshold(node):
                if (thresholds := node.inputs[1]) is None:
                    continue

                if (weights := node.inputs[2]) is None:
                    continue

                if thresholds := ir.convenience.get_const_tensor(thresholds):
                    if weights := ir.convenience.get_const_tensor(weights):
                        # Expand the step weights to match the thresholds as the
                        # weights are allowed to be scalar
                        thresholds = thresholds.numpy()
                        weights = weights.numpy()

                        weights = np.broadcast_to(
                            weights,
                            np.broadcast_shapes(
                                weights.shape,
                                thresholds.shape
                            )
                        )

                        # Sort steps into increasing order of thresholds, such
                        # that the cumulative sum over weights describes the
                        # output levels of the function.
                        order = np.argsort(thresholds, axis=-1)
                        weights = np.take_along_axis(weights, order, axis=-1)

                        levels = np.cumsum(np.pad(weights, {-1: (1, 0)}), -1)
                        levels = np.astype(levels, np.float64)

                        multithreshold_range.per_node[node.name] = float(
                            np.max(levels) - np.min(levels)
                        )

        return multithreshold_range


class MultiThresholdMonotonicity(BaseModel, use_attribute_docstrings=True):
    """Monotonicity characterization of MultiThreshold operators in the model.

    Monotonicity is characterized by counting the number of increasing and
    decreasing segments of each MultiThreshold function.

    A MultiThreshold function is considered non-monotonic if it has increasing
    and decreasing segments.
    """

    increasing_segments: dict[str | None, int] = {}
    """Number of increasing segments per MultiThreshold in the model."""

    decreasing_segments: dict[str | None, int] = {}
    """Number of decreasing segments per MultiThreshold in the model."""

    @computed_field
    @property
    def monotonic(self) -> int:
        """Number of monotonic MultiThreshold functions in the model."""

        count = 0

        for node, segments in self.increasing_segments.items():
            if segments <= 0 or self.decreasing_segments[node] <= 0:
                count += 1

        return count

    @classmethod
    def from_model(cls, model: ir.Model) -> "MultiThresholdMonotonicity":
        """Collect statistics from model."""

        monotonicity = MultiThresholdMonotonicity()

        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            if is_multithreshold(node):
                if (thresholds := node.inputs[1]) is None:
                    continue

                if (weights := node.inputs[2]) is None:
                    continue

                if thresholds := ir.convenience.get_const_tensor(thresholds):
                    if weights := ir.convenience.get_const_tensor(weights):
                        # Expand the step weights to match the thresholds as the
                        # weights are allowed to be scalar
                        thresholds = thresholds.numpy()
                        weights = weights.numpy()

                        weights = np.broadcast_to(
                            weights,
                            np.broadcast_shapes(
                                weights.shape,
                                thresholds.shape
                            )
                        )

                        # Sort steps into increasing order of thresholds, such
                        # that the shifted difference highlights the boundaries
                        # between monotonic segments.
                        order = np.argsort(thresholds, axis=-1)
                        weights = np.take_along_axis(weights, order, axis=-1)
                        weights = np.astype(weights, np.float64)

                        weights = np.pad(weights, {-1: (1, 0)})
                        difference = weights[..., 1:] - weights[..., :-1]

                        monotonicity.increasing_segments[node.name] = int(
                            np.max(np.sum(difference > 0, axis=-1))
                        )

                        monotonicity.decreasing_segments[node.name] = int(
                            np.max(np.sum(difference < 0, axis=-1))
                        )

        return monotonicity


class MultiThresholdSummary(BaseModel, use_attribute_docstrings=True):
    """Summary statistics of MultiThreshold operators in the model."""

    steps: MultiThresholdSteps = MultiThresholdSteps()
    """Steps of MultiThreshold operators."""

    depth: MultiThresholdDepth = MultiThresholdDepth()
    """Depth of MultiThreshold operators."""

    range: MultiThresholdRange = MultiThresholdRange()
    """Range of MultiThreshold operators."""

    monotonicity: MultiThresholdMonotonicity = MultiThresholdMonotonicity()
    """Monotonicity characterization of MultiThreshold operators."""

    total: int
    """Total number of MultiThreshold operators in the model."""

    @classmethod
    def from_model(cls, model: ir.Model) -> "MultiThresholdSummary":
        """Collect statistics from model."""

        summary = MultiThresholdSummary(total=0)

        summary.steps = MultiThresholdSteps.from_model(model)
        summary.depth = MultiThresholdDepth.from_model(model)
        summary.range = MultiThresholdRange.from_model(model)

        summary.monotonicity = MultiThresholdMonotonicity.from_model(model)

        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            if is_multithreshold(node):
                summary.total += 1

        return summary
