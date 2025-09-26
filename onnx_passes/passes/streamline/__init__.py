# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# Include basic streamlining transformations
import onnx_passes.passes.streamline.algebraic
import onnx_passes.passes.streamline.shapes
import onnx_passes.passes.streamline.transpose
import onnx_passes.passes.streamline.thresholds
import onnx_passes.passes.streamline.slice


# Set of so-called "streamlining" transformations: Moves scales and biases
# through the model graph and tries to collapse them via constant folding
@passes.verify.tolerance
@passes.register("streamline")
class Streamline(passes.compose.ComposePass, passes.base.Transformation):
    # Ordered sequence of passes and pass collections to be applied for each
    # iteration of streamlining
    __passes__ = [
        # Core of streamlining: Rearranging operators and grouping constants and
        # non-constants, such as scales and biases
        "algebraic",
        # Core of streamlining: Rearranging operators related to shape and
        # layout transformations, such as Reshape and Transpose
        "streamline-shapes",
        # After rearranging the graph, make sure everything is properly
        # annotated with shapes
        "shape-inference",
        # After grouping constants, more of them should be foldable into
        # Constant operators
        "fold-constants",
        # Now there could be constants and identities to get rid of, such as
        # adding a zero tensor
        "eliminate",
        # Finally cleanup the graph by removing nodes, constants, attributes,
        # etc. no longer needed and bringing it into an ordered state
        "cleanup"
    ]

    # Keep iterating the streamlining passes until the model stops changing
    __exhaustive__ = True
