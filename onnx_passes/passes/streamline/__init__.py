# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes


# Set of so-called "streamlining" transformations: Moves scales and biases
# through the model graph and tries to collapse them via constant folding
@passes.register("streamline")
class Streamline(passes.composed.ComposedPass, passes.base.Transformation):
    __passes__ = ["shape-inference", "associative", "fold-constants", "cleanup"]
    __exhaustive__ = True
