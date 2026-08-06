from onnx_passes.passes._base import RewriteRule
from onnx_passes.passes._verify import Verify

import onnx_ir as ir
import numpy as np


class RewriteFlattenAsReshape_v1(RewriteRule, Verify):
    """Rewrite Flatten operations as Reshape."""

    @staticmethod
    def pattern_v13(op, x, axis):
        return op.Flatten(x, axis=axis)

    @staticmethod
    def rewrite_v13(op, x, axis):
        # Calculate the sizes of the two output axes: The first flattens the
        # input shape up to the axis, the second flattens the rest.
        dim0 = op.ReduceProd(
            # ONNX < v15 does not support sliced op.Shape, rewrite as op.Slice
            op.Slice(
                op.Shape(x),
                op.Constant(value_ints=[0]),
                op.Constant(value_ints=[axis.as_int()])
            )
        )

        dim1 = op.Div(
            op.ReduceProd(op.Shape(x)),
            dim0
        )

        return op.Reshape(x, op.Concat(dim0, dim1, axis=0))

    @staticmethod
    def rewrite_v15(op, x, axis):
        # Calculate the sizes of the two output axes: The first flattens the
        # input shape up to the axis, the second flattens the rest.
        return op.Reshape(
            x,
            op.Concat(
                # ONNX >= v15 supports sliced op.Shape
                op.ReduceProd(op.Shape(x, end=axis.as_int())),
                op.ReduceProd(op.Shape(x, start=axis.as_int())),
                axis=0
            )
        )


class InferSqueezeAxes_v1(RewriteRule, Verify):
    """Infers axes to squeeze if no axes input or attribute is given."""

    @staticmethod
    def pattern(op, x):
        return op.Squeeze(x, _outputs=["y"])

    @staticmethod
    def check_v11(op, x, y):
        if y.producer().attributes.get("axes") is None:
            return x.shape is not None and x.shape.is_static()
        return False

    @staticmethod
    def rewrite_v11(op, x, y):
        return op.Squeeze(
            # <= v11 axes are specified as an attribute
            x, axes=np.where(np.asarray(x.shape) == 1)[0].tolist()
        )

    @staticmethod
    def rewrite_v13(op, x, y):
        return op.Squeeze(
            x,
            # >= v13 axes are specified as an input
            op.Reshape(
                op.NonZero(op.Equal(op.Shape(x), op.Constant(value_int=1))),
                op.Constant(value_ints=[-1])
            )
        )


class RewriteSqueezeAsReshape_v1(RewriteRule, Verify):
    """Rewrite Squeeze operations as Reshape."""

    @staticmethod
    def pattern_v11(op, x, axes):
        # <= v11 axes are specified as an attribute
        return op.Squeeze(x, axes=axes)

    @staticmethod
    def pattern_v13(op, x, axes):
        # >= v13 axes are specified as an input
        return op.Squeeze(x, axes)

    @staticmethod
    def rewrite_v11(op, x, axes):
        return RewriteSqueezeAsReshape_v1.rewrite_v13(
            op, x, op.Constant(value_ints=axes.as_ints())
        )

    @staticmethod
    def rewrite_v13(op, x, axes):
        # Mark axes selected to be squeezed by negative sizes (these cannot
        # appear as the output of op.Shape by default)
        shape = op.ScatterElements(
            op.Shape(x),
            axes,
            op.Expand(op.Constant(value_int=-1), op.Shape(axes))
        )

        # Generate indices of all entries from the input shape which are not
        # marked by -1, i.e., those entries to keep
        keep = op.NonZero(op.Not(op.Equal(shape, op.Constant(value_int=-1))))

        # Select all entries from the input shape to keep after getting rid of
        # some extra dimension inserted by op.NonZero
        shape = op.Gather(
            op.Shape(x), op.Reshape(keep, op.Constant(value_ints=[-1]))
        )

        # Use the (dynamic) shape calculation as second input to the reshape
        # operation finally replacing the squeeze
        return op.Reshape(x, shape)


class RewriteUnsqueezeAsReshape_v1(RewriteRule, Verify):
    """Rewrite Unsqueeze operations as Reshape."""

    @staticmethod
    def pattern_v11(op, x, axes):
        # <= v11 axes are specified as an attribute
        return op.Unsqueeze(x, axes=axes)

    @staticmethod
    def pattern_v13(op, x, axes):
        # >= v13 axes are specified as an input
        return op.Unsqueeze(x, axes)

    @staticmethod
    def rewrite_v11(op, x, axes):
        return RewriteUnsqueezeAsReshape_v1.rewrite_v13(
            op, x, op.Constant(value_ints=axes.as_ints())
        )

    @staticmethod
    def rewrite_v13(op, x, axes):
        # All zero and all one tensors covering the axes used for repeatedly
        # updating the indices and shape calculated below
        _0 = op.Expand(op.Constant(value_int=0), op.Shape(axes))
        _1 = op.Expand(op.Constant(value_int=1), op.Shape(axes))

        # The rank of the unsqueezed output: Old rank + inserted dimensions
        rank = op.Add(op.Size(op.Shape(x)), op.Size(axes))

        # Start operating on a sequence of indices mapping from new to old
        # dimensions: Seed mapping to 1-based indexing...
        indices = op.ConstantOfShape(
            op.Reshape(rank, op.Constant(value_ints=[-1])),
            value=ir.tensor([1])
        )

        # Update the index mapping by (1) skipping the unsqueezed dimensions,
        # (2) cumulatively adding up the input dimensions and, (3) subtracting
        # one to move to a zero-based indexing
        indices = op.Sub(
            op.CumSum(
                op.ScatterElements(indices, axes, _0),
                op.Constant(value_int=0)
            ),
            op.Constant(value_int=1)
        )

        # Derive the output shape by (1) collecting input dimensions according
        # to the index mapping and, (2) updating the shape by setting all
        # unsqueezed dimension to 1
        shape = op.ScatterElements(op.Gather(op.Shape(x), indices), axes, _1)

        # Use the (dynamic) shape calculation as second input to the reshape
        # operation finally replacing the unsqueeze
        return op.Reshape(x, shape)


class RewriteExpandAsReshape_v1(RewriteRule, Verify):
    """Rewrite Expand operations as Reshape if expand effectively unsqueezes."""

    @staticmethod
    def pattern(op, x, shape):
        return op.Expand(x, shape)

    @staticmethod
    def check(op, x, shape):
        if (shape := ir.convenience.get_const_tensor(shape)) is not None:
            if x.shape is not None and x.shape.is_static():
                return np.prod(shape.numpy()) == np.prod(x.shape.numpy())
        return False

    @staticmethod
    def rewrite(op, x, shape):
        return op.Reshape(x, shape)


class FuseConsecutiveReshapes_v1(RewriteRule, Verify):
    """Fuses two consecutive reshape operations into a single reshape."""

    @staticmethod
    def pattern(op, x, shape1, shape2):
        return op.Reshape(op.Reshape(x, shape1), shape2, _outputs=["y"])

    @staticmethod
    def rewrite_v5(op, x, shape1, shape2, y):
        return op.Reshape(x, shape2)

    @staticmethod
    def rewrite_v14(op, x, shape1, shape2, y):
        # Default allowzero according to ONNX operators reference documentation:
        #   https://onnx.ai/onnx/operators/onnx__Reshape.html#reshape-14
        if not (allowzero := y.producer().attributes.get("allowzero", None)):
            allowzero = ir.Attr("allowzero", ir.AttributeType.INT, 0)

        # Start by assuming the shape of the second reshape to fully determine
        # the final output shape, which is almost always the case
        shape = shape2

        # Turn allowzero=0 pass-through dimensions of the second reshape into
        # explicit dimensions inferred from the shape of the first reshape
        if allowzero is None or allowzero.as_int() == 0:
            # Find indices of dimensions to be passed through from the shape of
            # the first reshape, i.e., those where the second shape has zeros
            i = op.Reshape(
                op.NonZero(op.Equal(shape2, op.Constant(value_int=0))),
                op.Constant(value_ints=[-1])
            )

            # Update the output shape with pass-through entries gathered from
            # the intermediate shape
            shape = op.ScatterElements(shape2, i, op.Gather(shape1, i))

        # Fused reshape keeping the allowzero attribute of the second reshape
        return op.Reshape(
            x, shape, allowzero=allowzero.as_int()  # noqa: allowzero not None
        )
