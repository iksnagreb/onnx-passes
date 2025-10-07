# ir.Value, ir.Attr, ir.tensor
import onnx_ir as ir

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# Derive Transformations (allowed to modify the graph) from pattern-based
# rewrite rules
from onnx_passes.passes.base import Transformation, RewriteRuleSetPass
# Collecting node attributes with optional defaults
from onnx_passes.passes.util import collect_attrs

# Domain used by custom operators implemented with this library
from onnx_passes.ops import DOMAIN as CUSTOM_DOMAIN

# Numpy for index and shape calculations
import numpy as np


# Alias to save some brackets...
def prod(*args):
    return np.prod([*args])


# Lowers convolution operators to MatMul (plus input generator Im2Col). The
# optional bias is extracted as a standalone operator and grouped convolutions
# are implemented via Split and Concat operators.
@passes.register("lower-conv")
@passes.verify.tolerance
class ConvToMatMul(Transformation, RewriteRuleSetPass):
    def pattern(self):
        return [
            lambda op, x, w, b: op.Conv(x, w, b, _outputs=["y"]),
            lambda op, x, w: op.Conv(x, w, _outputs=["y"]),
        ]

    def check(self):
        # Check handling both pattern alternatives checking for the node bing in
        # a valid state
        def _check(op, x, w, y, b=None):
            # Inputs and output must have shape annotations to carry out
            # extensive shape calculations below
            if x.shape is None or y.shape is None or w.shape is None:
                return False

            # Decompose input, output and weight tensor shapes to get input and
            # output channels, kernel sizes and feature map sizes
            N, C, *Is = x.shape  # noqa: Duplicate extracting shapes and attrs
            _, M, *Os = y.shape
            m, c, *Ks = w.shape

            # Default attributes of the Conv operator according to operators
            # reference: https://onnx.ai/onnx/operators/onnx__Conv.html
            attributes = {
                "auto_pad": (ir.AttributeType.STRING, "NOTSET"),
                "dilations": (ir.AttributeType.INTS, [1 for _ in Is]),
                "group": (ir.AttributeType.INT, 1),
                "kernel_shape": (ir.AttributeType.INTS, [*Ks]),
                "pads": (ir.AttributeType.INTS, [0 for _ in [*Is, *Is]]),
                "strides": (ir.AttributeType.INTS, [1 for _ in Is]),
            }

            # Collect node attributes falling back to defaults defined above
            attributes = collect_attrs(y.producer(), attributes)

            # If auto padding is set to VALID or any of the SAME_* paddings,
            # there cannot be explicit padding amounts
            if attributes["auto_pad"].as_string() != "NOTSET":
                if np.any(np.asarray(attributes["pads"].as_ints()) != 0):
                    return False

            # Group cannot be zero or negative
            if attributes["group"].as_int() <= 0:
                return False

            # The kernel shape must match the weight tensor shape
            if np.any(np.asarray(attributes["kernel_shape"].as_ints()) != Ks):
                return False

            # Operator is in a valid state and can be accepted for rewriting
            return True

        return [
            lambda op, x, w, b, y: _check(op, x, w, y, b),
            lambda op, x, w, y: _check(op, x, w, y),
        ]

    def rewrite(self):
        # Replacement handling both pattern alternatives depending on whether
        # the optional input b has been matched
        def _rewrite(op, x, w, y, b=None):
            # Decompose input, output and weight tensor shapes to get input and
            # output channels, kernel sizes and feature map sizes
            N, C, *Is = x.shape  # noqa: Duplicate extracting shapes and attrs
            _, M, *Os = y.shape
            m, c, *Ks = w.shape

            # Default attributes of the Conv operator according to operators
            # reference: https://onnx.ai/onnx/operators/onnx__Conv.html
            attributes = {
                "auto_pad": (ir.AttributeType.STRING, "NOTSET"),
                "dilations": (ir.AttributeType.INTS, [1 for _ in Is]),
                "group": (ir.AttributeType.INT, 1),
                "kernel_shape": (ir.AttributeType.INTS, [*Ks]),
                "pads": (ir.AttributeType.INTS, [0 for _ in [*Is, *Is]]),
                "strides": (ir.AttributeType.INTS, [1 for _ in Is]),
            }

            # Collect node attributes falling back to defaults defined above
            attributes = collect_attrs(y.producer(), attributes)

            strides = attributes["strides"].as_ints()
            dilations = attributes["dilations"].as_ints()
            pads = attributes["pads"].as_ints()

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

            # As padding is now explicit, the attribute can be deleted, there is
            # no reason to forward this to the input generator
            del attributes["auto_pad"]

            # Insert explicit padding operator at the input of the whole
            # operator pattern (omit if all paddings are zero)
            if np.any(pads):
                x = op.Pad(
                    x,
                    # Convert the padding per axis from attribute to constant
                    # tensor
                    op.Constant(value=ir.tensor(pads)),
                    # Make implicit zero padding explicit
                    op.Constant(value=ir.tensor(0.0, x.dtype)),
                    # Padding along spatial axes only
                    op.Constant(
                        value=ir.tensor([i + 2 for i in range(len(Is))])
                    )
                )

            # As padding is inserted as a standalone Pad operator, remove the
            # padding attribute from the input generator
            del attributes["pads"]

            # Update the input shape after padding as all calculations below
            # assume the padded shape
            for i, (size, lower, upper) in enumerate(
                    zip(Is, pads[:len(Is)], pads[len(Is):])
            ):
                Is[i] = size + lower + upper

            # Convert from channels-first layout to channels-last layout by
            # transposing the axes
            x = op.Transpose(x, perm=[0, *[i + 2 for i in range(len(Is))], 1])

            # Number of groups for grouped convolution: Split operator long the
            # channel axis in groups
            groups = attributes["group"].as_int()

            # Split the input tensor into groups along the channel axis for
            # parallel convolution groups: C input channels
            xs = op.Split(
                x, op.Constant(
                    value=ir.tensor([C // groups for _ in range(groups)])
                ), axis=-1, _outputs=groups
            )

            # Transpose weight tensor to channel-last layout on the input side,
            # keeping the M output channels first
            w = op.Transpose(w, perm=[0, *[i + 2 for i in range(len(Ks))], 1])

            # Flatten the kernel dimensions of the weight tensor into the input
            # channel axis
            w = op.Reshape(w, op.Constant(value_ints=[M, int(c * np.prod(Ks))]))

            # Transpose output and input axis of the weight matrix as this will
            # end up on the right hand side of the MatMul, i.e., computing xW^T
            w = op.Transpose(w)

            # Split the weight tensor into groups along the channel axis for
            # parallel convolution groups: M output channels
            ws = op.Split(
                w, op.Constant(
                    value=ir.tensor([M // groups for _ in range(groups)])
                ), axis=1, _outputs=groups
            )

            # Wrap splits in list if there are no groups to simplify the pattern
            # generation below (allows to use the loop even for no groups)
            xs, ws = ([xs], [ws]) if groups == 1 else (xs, ws)

            # Remove the group attribute as this is handled by rewriting the
            # pattern into parallel convolution branches and the input generator
            # reusing the attributes does not handle any grouping
            del attributes["group"]

            # Generate indices for each output element with expanded kernel
            # dimensions at the end
            i = np.unravel_index(np.arange(prod(*Os, *Ks, c)), [*Os, *Ks, c])

            # Decompose multi-index into spatial, kernel and channel sections
            spatial, kernel, cs = i[0:len(Os)], i[len(Os):-1], i[-1]

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
            j = np.ravel_multi_index(j, [*Is, c]).reshape([*Os, prod(*Ks, c)])

            # Insert precomputed indices as a constant into the graph
            j = op.Constant(value=ir.tensor(j))

            # Collect replacement patterns for parallel branches of convolution
            # groups
            ys = []

            # Rewrite pattern for each convolution group receiving the
            # corresponding split of the input and weights tensor
            for x, weight in zip(xs, ws):
                # # Insert explicit padding operator at the input of the current
                # # convolution group (omit if all paddings are zero)
                # # TODO: Consider inserting padding at the group level? Maybe
                # #  make this configurable?
                # if np.any(pads):
                #     x = op.Pad(
                #         x,
                #         # Convert the padding per axis from attribute to
                #         # constant tensor
                #         op.Constant(value=ir.tensor(pads)),
                #         # Make implicit zero padding explicit
                #         op.Constant(value=ir.tensor(0.0, x.dtype)),
                #         # Padding along spatial axes only
                #         op.Constant(
                #             value=ir.tensor([i + 1 for i in range(len(Is))])
                #         )
                #     )

                # Im2Col receives both attributes and precomputed indices: Pure
                # ONNX uses the precomputed indices to gather the sliding window
                # while downstream transformations could operate on the
                # attributes
                im2col = op.Im2Col(x, j, **attributes, _domain=CUSTOM_DOMAIN)

                # Replace each convolution by input generator followed by MatMul
                ys.append(op.MatMul(im2col, weight))

            # Join the parallel convolution branches along the channel axis
            y = op.Concat(*ys, axis=-1)

            # If the optional bias input is present, expand the bias for
            # broadcasting to the feature map dimensions and add to the output
            if b is not None:
                y = op.Add(y, b)

            # Convert from channels-last back to channels-first layout by
            # transposing the axes
            return op.Transpose(
                y, perm=[0, len(Os) + 1, *[i + 1 for i in range(len(Os))]]
            )

        return [
            lambda op, x, w, b, y: _rewrite(op, x, w, y, b),
            lambda op, x, w, y: _rewrite(op, x, w, y),
        ]
