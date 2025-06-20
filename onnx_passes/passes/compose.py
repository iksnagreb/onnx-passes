# The base classes defined below are still not fully functional passes, but
# abstract bases themselves
import abc

# ir.Model, ir.save, ...
import onnx_ir as ir

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work, passes.collect and passes.base.Pass
import onnx_passes.passes as passes


# Composes a list of passes to a single pass which can optionally be applied
# exhaustively
class ComposePass(passes.base.Pass, abc.ABC):
    # Sequence of passes as class attribute - will be parsed to Pass instances
    # when initializing the concrete pass instance
    __passes__: list[str]
    # Mark the composed sequence of passes to be applied exhaustively until the
    # model stops changing
    __exhaustive__: bool = False

    # Initializes a pass sets references to optional configuration and state
    # dictionary as instance attributes
    def __init__(self, config: dict | None, state: dict | None):
        # Initialize the pass base class connecting to the configuration and
        # state dictionary
        super().__init__(config=config, state=state)
        # Collect and instantiate all ONNX IR passes from the sequence by name
        # and connect each pass to the shared configuration and state dictionary
        self.passes = ir.passes.PassManager([
            cls(config, state) for cls in passes.collect(self.__passes__)
        ])

    # Runs the sequences of composed passes on the ONNX model
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Always apply the passes at least once
        result = self.passes(model)
        # If the composed pass is marked exhaustive, apply the sequence of
        # passes as long as there are changes to the model
        while self.__exhaustive__ and result.modified:
            result = self.passes(result.model)
        # Return the final result of the composed passes
        return result
