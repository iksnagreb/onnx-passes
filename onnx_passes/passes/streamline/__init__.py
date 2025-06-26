# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# Include basic streamlining transformations
import onnx_passes.passes.streamline.algebraic


# Set of so-called "streamlining" transformations: Moves scales and biases
# through the model graph and tries to collapse them via constant folding
@passes.verify.tolerance
@passes.register("streamline")
class Streamline(passes.compose.ComposePass, passes.base.Transformation):
    __passes__ = ["algebraic", "shape-inference", "fold-constants", "cleanup"]
    __exhaustive__ = True
