# ir.Value, ir.Attr, ir.tensor
import onnx_ir as ir

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# Derive Transformations (allowed to modify the graph) from pattern-based
# rewrite rules
from onnx_passes.passes.base import Transformation, RewriteRulePass
# Collecting node attributes with optional defaults, generating constants of 1
# matching another tensor in shape and type
from onnx_passes.passes.util import collect_attrs, ones_like

# Domain used by custom operators implemented with this library
from onnx_passes.ops import DOMAIN as CUSTOM_DOMAIN
# Make custom Im2Col operator available for lowering pooling
from onnx_passes.ops.im2col import Im2Col  # noqa: Used indirectly via registry

# Numpy for index and shape calculations
import numpy as np


# Alias to save some brackets...
def prod(*args):
    return int(np.prod([*args]))


# Turns global average pooling, i.e., kernel implicitly spanning the whole
# feature map, into average pooling with explicit kernel size.
@passes.verify.tolerance
@passes.register("lower-pooling")
class ExplicitGlobalAveragePool(Transformation, RewriteRulePass):
    def pattern(self, op, x):
        return op.GlobalAveragePool(x)

    def check(self, op, x):
        if x.shape is None or not x.shape.is_static():
            return False
        return True

    def rewrite(self, op, x):
        # TODO: Alternatively, this could be directly lowered to ReduceMean over
        #  all trailing axes (should even work without static shape)
        return op.AveragePool(x, count_include_pad=1, kernel_shape=x.shape[2:])


# Turns global max pooling, i.e., kernel implicitly spanning the whole feature
# map, into max pooling with explicit kernel size.
@passes.verify.tolerance
@passes.register("lower-pooling")
class ExplicitGlobalMaxPool(Transformation, RewriteRulePass):
    def pattern(self, op, x):
        return op.GlobalMaxPool(x)

    def check(self, op, x):
        if x.shape is None or not x.shape.is_static():
            return False
        return True

    def rewrite(self, op, x):
        # TODO: Alternatively, this could be directly lowered to ReduceMax over
        #  all trailing axes (should even work without static shape)
        return op.MaxPool(x, kernel_shape=x.shape[2:])


# Turns global lp-pooling, i.e., kernel implicitly spanning the whole feature
# map, into lp-pooling with explicit kernel size.
@passes.verify.tolerance
@passes.register("lower-pooling")
class ExplicitGlobalLpPool(Transformation, RewriteRulePass):
    def pattern(self, op, x):
        return op.GlobalLpPool(x, _outputs=["y"])

    def check(self, op, x, y):
        if x.shape is None or not x.shape.is_static():
            return False
        return True

    def rewrite(self, op, x, y: ir.Value):
        # Default attributes of the pooling operator according to operators
        # reference: https://onnx.ai/onnx/operators/onnx__GlobalLpPool.html
        if (p := y.producer().attributes.get("p")) is None:
            p = ir.Attr("p", ir.AttributeType.INT, 2)
        # Replacement always with explicit norm p
        return op.LpPool(x, p=p, kernel_shape=x.shape[2:])


# Generates the full index pattern for sliding window generator (without padding
# and grouping)
def _make_im2col_indices(outs, ins, c, ks, strides, dilations):
    # Generate indices for each output element with expanded kernel
    # dimensions at the end
    i = np.unravel_index(np.arange(prod(*outs, *ks, c)), [*outs, *ks, c])

    # Decompose multi-index into spatial, kernel and channel sections
    spatial, kernel, cs = i[0:len(outs)], i[len(outs):-1], i[-1]

    # Collect multi-index into the input as a list: This will cover
    # spatial and channel axes, there are no kernel axes in the input
    j = []

    # For ech spatial dimension there is a corresponding kernel index
    # and stride and dilation relating the output to the input index
    for ii, kk, s, d in zip(spatial, kernel, strides, dilations):
        # Stride determines the spatial offset and dilation the kernel
        # offset - together this is the input spatial index
        j.append(s * ii + d * kk)

    # Add the channel indices at the end: The pure channel component is
    # the same between input and output as this is the last or innermost
    # dimension
    j.append(cs)

    # Convert the multi-index list to flat indices into the input tensor
    # with the shape matching the output tensor (flattened kernel and
    # channel dimensions) - using this to gather from the flattened
    # input tensor expands to the expected shape without extra Reshape.
    return np.ravel_multi_index(j, [*ins, c]).reshape([*outs, prod(*ks, c)])


# Lowers MaxPool to ReduceMax (plus Im2Col and reshaping and transposing of
# inputs)
@passes.register("lower-pooling")
@passes.verify.tolerance
class MaxPoolToReduceMax(Transformation, RewriteRulePass):
    def pattern(self, op, x):
        # TODO: Extend to match the optional index output as well...
        return op.MaxPool(x, _outputs=["y"])

    def check(self, op, x, y):
        # Inputs and output must have shape annotations to carry out extensive
        # shape calculations below
        if x.shape is None or not x.shape.is_static():
            return False

        if y.shape is None or not y.shape.is_static():
            return False

        # Default attributes of the max pooling operator according to operators
        # reference: https://onnx.ai/onnx/operators/onnx__MaxPool.html
        attributes = {
            "auto_pad": (ir.AttributeType.STRING, "NOTSET"),
        }

        # Collect node attributes falling back to defaults defined above
        attributes = collect_attrs(y.producer(), attributes)

        # If auto padding is set to VALID or any of the SAME_* paddings,
        # there cannot be explicit padding amounts
        if attributes["auto_pad"].as_string() != "NOTSET":
            if np.any(np.asarray(attributes["pads"].as_ints()) != 0):
                return False

        # Operator is in a valid state and can be accepted for rewriting
        return True

    def rewrite(self, op, x, y):
        # Decompose input and output tensor shapes to get input and output
        # channels and feature map sizes
        N, C, *Is = x.shape
        _, M, *Os = y.shape

        # Default attributes of the max pooling operator according to operators
        # reference: https://onnx.ai/onnx/operators/onnx__MaxPool.html
        attributes = {
            "auto_pad": (ir.AttributeType.STRING, "NOTSET"),
            "ceil_mode": (ir.AttributeType.INT, 0),
            "dilations": (ir.AttributeType.INTS, [1 for _ in Is]),
            "kernel_shape": (ir.AttributeType.INTS, None),  # required!
            "pads": (ir.AttributeType.INTS, [0 for _ in [*Is, *Is]]),
            "storage_order": (ir.AttributeType.INT, 0),
            "strides": (ir.AttributeType.INTS, [1 for _ in Is]),
        }

        # Collect node attributes falling back to defaults defined above
        attributes = collect_attrs(y.producer(), attributes)

        strides = attributes["strides"].as_ints()
        dilations = attributes["dilations"].as_ints()
        pads = attributes["pads"].as_ints()
        Ks = attributes["kernel_shape"].as_ints()

        # Make SAME_* padding explicit before separating out into a
        # standalone padding operator
        if attributes["auto_pad"].as_string().startswith("SAME"):
            # Pads per dimension such that the output has the same size as
            # the input (actually output = ceil(input / stride))
            pads = [d * (k - 1) for d, k in zip(dilations, Ks)]
            # Distribute pads to beginning and end with uneven amounts
            # distributed according to the SAME_* attribute
            pads = {
                # Extra padding goes to the beginning
                "SAME_LOWER": [
                    *[np.ceil(n / 2) for n in pads], *[n // 2 for n in pads]
                ],
                # Extra padding goes to the end
                "SAME_UPPER": [
                    *[n // 2 for n in pads], *[np.ceil(n / 2) for n in pads]
                ]
            }[attributes["auto_pad"].as_string()]
            # Make sure all pads are simple python integers...
            pads = [int(n) for n in pads]

        # Match condition ensures that padding is defined consistently, so
        # it is safe to overwrite the attribute
        attributes["pads"] = ir.Attr("pads", ir.AttributeType.INTS, pads)

        # As padding is now explicit, the attribute can be deleted, there is no
        # reason to forward this to the input generator
        del attributes["auto_pad"]

        # Insert explicit padding operator at the input of the whole operator
        # pattern (omit if all paddings are zero)
        if np.any(pads):
            x = op.Pad(
                x,
                # Convert the padding per axis from attribute to constant tensor
                op.Constant(value=ir.tensor(pads)),
                # Make implicit negative infinity padding explicit
                op.Constant(value=ir.tensor(-np.inf, x.dtype)),
                # Padding along spatial axes only
                op.Constant(value=ir.tensor([i + 2 for i in range(len(Is))]))
            )

        # As padding is inserted as a standalone Pad operator, remove the
        # padding attribute from the input generator
        del attributes["pads"]

        # Update the input shape after padding as all calculations below assume
        # the padded shape
        for i, (size, lower, upper) in enumerate(
                zip(Is, pads[:len(Is)], pads[len(Is):])
        ):
            Is[i] = size + lower + upper

        # Get rid of the ceiling mode and the storage order which are not used
        # by the lowered operator
        del attributes["ceil_mode"]
        del attributes["storage_order"]

        # Convert from channels-first layout to channels-last layout by
        # transposing the axes
        x = op.Transpose(x, perm=[0, *[i + 2 for i in range(len(Is))], 1])

        # Precompute the sliding window generator access pattern to insert as a
        # constant into the graph for executing the ONNX reference of Im2Col
        j = op.Constant(value=ir.tensor(
            _make_im2col_indices(Os, Is, C, Ks, strides, dilations)
        ))

        # Im2Col receives both attributes and precomputed indices: Pure ONNX
        # uses the precomputed indices to gather the sliding window while
        # downstream transformations could operate on the attributes.
        y = op.Im2Col(x, j, **attributes, _domain=CUSTOM_DOMAIN)

        # Reshape to disentangle the channel and kernel dimensions as pooling
        # does not reduce along the channel axis, but Im2Col packs both into a
        # single axis (the last one)
        y = op.Reshape(y, op.Constant(value_ints=[N, *Os, prod(*Ks), C]))

        # Reduce over the flattened kernel dimensions (now second to last axis)
        # and get rid of this axis to have the expected rank at the output
        y = op.ReduceMax(y, op.Constant(value_ints=[len(Os) + 1]), keepdims=0)

        # Convert from channels-last back to channels-first layout transposing
        # the axes
        return op.Transpose(
            y, perm=[0, len(Os) + 1, *[i + 1 for i in range(len(Os))]]
        )


# Lowers AveragePool to ReduceSum (plus Im2Col and reshaping and transposing of
# inputs)
#
# Note: As this expands the averaging scale to the full output size, this should
# be combined with unbroadcasting to remove any redundancies.
@passes.register("lower-pooling")
@passes.verify.tolerance
class AveragePoolToReduceSum(Transformation, RewriteRulePass):
    def pattern(self, op, x):
        return op.AveragePool(x, _outputs=["y"])

    def check(self, op, x, y):
        # Inputs and output must have shape annotations to carry out extensive
        # shape calculations below
        if x.shape is None or not x.shape.is_static():
            return False

        if y.shape is None or not y.shape.is_static():
            return False

        # Default attributes of the avg pooling operator according to operators
        # reference: https://onnx.ai/onnx/operators/onnx__AveragePool.html
        attributes = {
            "auto_pad": (ir.AttributeType.STRING, "NOTSET"),
            "count_include_pad": (ir.AttributeType.INT, 0)
        }

        # Collect node attributes falling back to defaults defined above
        attributes = collect_attrs(y.producer(), attributes)

        # If auto padding is set to VALID or any of the SAME_* paddings,
        # there cannot be explicit padding amounts
        if attributes["auto_pad"].as_string() != "NOTSET":
            if np.any(np.asarray(attributes["pads"].as_ints()) != 0):
                return False

        # This is an integer, but actually is interpreted as a boolean, it is
        # unclear how to interpret anything but 0 or 1, better reject this...
        if attributes["count_include_pad"].as_int() not in {0, 1}:
            return False

        # Operator is in a valid state and can be accepted for rewriting
        return True

    def rewrite(self, op, x, y):
        # Decompose input and output tensor shapes to get input and output
        # channels and feature map sizes
        N, C, *Is = x.shape
        _, M, *Os = y.shape

        # Default attributes of the max pooling operator according to operators
        # reference: https://onnx.ai/onnx/operators/onnx__MaxPool.html
        attributes = {
            "auto_pad": (ir.AttributeType.STRING, "NOTSET"),
            "ceil_mode": (ir.AttributeType.INT, 0),
            "count_include_pad": (ir.AttributeType.INT, 0),
            "dilations": (ir.AttributeType.INTS, [1 for _ in Is]),
            "kernel_shape": (ir.AttributeType.INTS, None),  # required!
            "pads": (ir.AttributeType.INTS, [0 for _ in [*Is, *Is]]),
            "strides": (ir.AttributeType.INTS, [1 for _ in Is]),
        }

        # Collect node attributes falling back to defaults defined above
        attributes = collect_attrs(y.producer(), attributes)

        strides = attributes["strides"].as_ints()
        dilations = attributes["dilations"].as_ints()
        pads = attributes["pads"].as_ints()
        Ks = attributes["kernel_shape"].as_ints()

        # Make SAME_* padding explicit before separating out into a
        # standalone padding operator
        if attributes["auto_pad"].as_string().startswith("SAME"):
            # Pads per dimension such that the output has the same size as
            # the input (actually output = ceil(input / stride))
            pads = [d * (k - 1) for d, k in zip(dilations, Ks)]
            # Distribute pads to beginning and end with uneven amounts
            # distributed according to the SAME_* attribute
            pads = {
                # Extra padding goes to the beginning
                "SAME_LOWER": [
                    *[np.ceil(n / 2) for n in pads], *[n // 2 for n in pads]
                ],
                # Extra padding goes to the end
                "SAME_UPPER": [
                    *[n // 2 for n in pads], *[np.ceil(n / 2) for n in pads]
                ]
            }[attributes["auto_pad"].as_string()]
            # Make sure all pads are simple python integers...
            pads = [int(n) for n in pads]

        # Match condition ensures that padding is defined consistently, so
        # it is safe to overwrite the attribute
        attributes["pads"] = ir.Attr("pads", ir.AttributeType.INTS, pads)

        # As padding is now explicit, the attribute can be deleted, there is no
        # reason to forward this to the input generator
        del attributes["auto_pad"]

        # Prepare counting elements for normalizing by dividing by the kernel
        # size. Each original element contributes 1 to the count.
        count = op.Cast(ones_like(op, x), to=ir.DataType.INT64)

        # Insert explicit padding operator at the input of the whole operator
        # pattern (omit if all paddings are zero)
        if np.any(pads):
            x = op.Pad(
                x,
                # Convert the padding per axis from attribute to constant tensor
                op.Constant(value=ir.tensor(pads)),
                # Make implicit zero padding explicit
                op.Constant(value=ir.tensor(0.0, x.dtype)),
                # Padding along spatial axes only
                op.Constant(value=ir.tensor([i + 2 for i in range(len(Is))]))
            )

            # Attribute controlling whether to include the padding in the
            # averaging calculation
            count_include_pad = attributes["count_include_pad"].as_int()

            # Counts should receive the same padding amount as the input, but,
            # depending on the attribute, these counts are considered for the
            # averaging calculation or not.
            count = op.Pad(
                count,
                # Convert the padding per axis from attribute to constant tensor
                op.Constant(value=ir.tensor(pads)),
                # Make implicit zero padding explicit
                op.Constant(value_int=count_include_pad),
                # Padding along spatial axes only
                op.Constant(value=ir.tensor([i + 2 for i in range(len(Is))]))
            )

        # As padding is inserted as a standalone Pad operator, remove the
        # padding attribute from the input generator
        del attributes["pads"]

        # Update the input shape after padding as all calculations below assume
        # the padded shape
        for i, (size, lower, upper) in enumerate(
                zip(Is, pads[:len(Is)], pads[len(Is):])
        ):
            Is[i] = size + lower + upper

        # Get rid of the ceiling mode which is not used by the lowered operator
        del attributes["ceil_mode"]

        # Permutations for converting between channels-first (implicitly assumed
        # by pooling) and channels-last (after lowering) layout
        to_channels_last = [0, *[i + 2 for i in range(len(Is))], 1]
        to_channels_first = [0, len(Os) + 1, *[i + 1 for i in range(len(Os))]]

        # Convert from channels-first layout to channels-last layout by
        # transposing the axes
        x = op.Transpose(x, perm=to_channels_last)

        # As counts are derived from the input, permute them to channels-last
        # layout as well
        count = op.Transpose(count, perm=to_channels_last)

        # Precompute the sliding window generator access pattern to insert as a
        # constant into the graph for executing the ONNX reference of Im2Col
        j = op.Constant(value=ir.tensor(
            _make_im2col_indices(Os, Is, C, Ks, strides, dilations)
        ))

        # Im2Col does not handle padding nor averaging, get rid of the
        # corresponding attribute
        del attributes["count_include_pad"]

        # Im2Col receives both attributes and precomputed indices: Pure ONNX
        # uses the precomputed indices to gather the sliding window while
        # downstream transformations could operate on the attributes.
        y = op.Im2Col(x, j, **attributes, _domain=CUSTOM_DOMAIN)

        # Reshape to disentangle the channel and kernel dimensions as pooling
        # does not reduce along the channel axis, but Im2Col packs both into a
        # single axis (the last one)
        y = op.Reshape(y, op.Constant(value_ints=[N, *Os, prod(*Ks), C]))

        # Reduce over the flattened kernel dimensions (now second to last axis)
        # and get rid of this axis to have the expected rank at the output
        # TODO: Change to ReduceSum followed by scale, precomputed to account
        #  for zero padding if count_include_pad=0
        y = op.ReduceSum(y, op.Constant(value_ints=[len(Os) + 1]), keepdims=0)

        # Likewise, generate sliding windows of the padded element counts for
        # each kernel and reduce these over the spatial dimensions to get the
        # element counts per kernel, which is the normalization scale.
        count = op.ReduceSum(
            op.Reshape(
                op.Im2Col(count, j, **attributes, _domain=CUSTOM_DOMAIN),
                op.Constant(value_ints=[N, *Os, prod(*Ks), C])
            ),
            op.Constant(value_ints=[len(Os) + 1]),
            keepdims=0
        )

        # Normalize by dividing by the element count per kernel
        y = op.Div(y, op.CastLike(count, y))

        # Convert from channels-last back to channels-first layout transposing
        # the axes
        return op.Transpose(y, perm=to_channels_first)

# TODO: LpPool seems confusing (use abs or not?, is p integer or float?) and
#  PyTorch does not actually seem to export LpPool but AveragePool...
