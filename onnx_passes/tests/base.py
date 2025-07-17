# Uses pytest for testing, test discovery and reporting
import pytest

# Dynamically import python modules at runtime used for dynamically registering
# passes according to configuration files
import importlib

# Passes are done in ONNX IR representation
import onnx_ir as ir

# Use ONNX Script for creating test models
from onnxscript import OnnxFunction
# ONNX script testing and verification infrastructure
from onnxscript.testing import assert_isomorphic, assert_onnx_proto_equal

# Collect custom ONNX IR passes from the library by name
from onnx_passes.passes import collect


# Asserts two ONNX protos are equal after ensuring both graphs have the same
# name. This is a workaround, as assert_isomorphic cannot compare initializers.
def _assert_onnx_proto_equal(model_1, model_2, **kwargs):
    model_1.graph.name = model_2.graph.name = "graph"
    assert_onnx_proto_equal(model_1, model_2, **kwargs)


# Base class for creating simple tests of passes: ...
class PassesTestBase:
    # List of passes to be tested: Supports full pass resolution from pass
    # class, full pass-class name, or category name
    __passes__: list = []

    # List of common passes applied to both, the input and expected model, for
    # performing cleanup from ONNX Script export: Supports full pass resolution
    __common__: list = ["cleanup", "checker", "verify"]

    # ONNX Script function for exporting the test model: Should be a method
    # decorated as @script and @staticmethod
    __model__: OnnxFunction = None

    # ONNX Script function for exporting the expected model: Should be a method
    # decorated as @script and @staticmethod
    __expected__: OnnxFunction = None

    # Function for generating input arrays for model verification: This is
    # optional, if there is no input generator, verification will be disabled
    __inputs__: callable = None

    # Shared configuration dictionary for all pass instances collected from the
    # __passes__ and __common__ lists
    __config__: dict = None

    # Shared state dictionary for all pass instances collected from the
    # __passes__ and __common__ lists - can be used to seed the state
    __state__: dict = None

    @property
    def reference(self):
        __tracebackhide__ = True

        # If there is an input generator function defined, generate inputs and
        # outputs via eager mode execution of the model
        if self.__inputs__ is not None:
            # If there is no model for testing, skip the test...
            if self.__model__ is None:
                pytest.skip(f"No __model__ for {self.__class__.__name__}")
            # Generate test inputs
            inputs = self.__inputs__()
            # Generate outputs by evaluating the model in eager mode, i.e., by
            # executing the python function
            outputs = self.__model__(*inputs)
            # Make sure outputs are always wrapped in a list
            if not isinstance(outputs, list | tuple):
                outputs = [outputs]
            # Return list of inputs and outputs
            return {"inp": inputs, "out": outputs}
        # Empty lists indicating no verification reference
        return {}

    @property
    def state(self):
        __tracebackhide__ = True

        state_dict = self.__state__ if self.__state__ is not None else {}

        if self.__inputs__ is not None:
            state_dict["reference"] = self.reference

        return state_dict

    @property
    def config(self):
        __tracebackhide__ = True

        config = self.__config__ if self.__config__ is not None else {}

        # Inject dynamic module imports if the configuration specifies an
        # imports section, e.g., for dynamically registering passes
        if "imports" in config:
            for name in config["imports"]:
                importlib.__import__(name)

        # Enable verification passes if not already enabled
        if self.__inputs__ is not None and "verify" not in config:
            config["verify"] = {True: True}

        return config

    @property
    def common(self):
        __tracebackhide__ = True

        common = [*self.config.setdefault("passes", []), *self.__common__]
        common = [cls(self.config, self.state) for cls in collect(common)]

        return ir.passes.PassManager(passes=common, steps=1)

    @property
    def passes(self):
        __tracebackhide__ = True

        if not self.__passes__:
            pytest.skip(f"No __passes__ for {self.__class__.__name__}")

        passes = [*self.__passes__, *self.__common__]
        passes = [cls(self.config, self.state) for cls in collect(passes)]

        return ir.passes.PassManager(passes=passes, steps=1)

    @property
    def model(self):
        __tracebackhide__ = True

        if self.__model__ is None:
            pytest.skip(f"No __model__ for {self.__class__.__name__}")

        result = self.common(ir.from_proto(self.__model__.to_model_proto()))
        return result.model

    @property
    def expected(self):
        __tracebackhide__ = True

        if self.__expected__ is None:
            pytest.skip(f"No __expected__ for {self.__class__.__name__}")

        result = self.common(ir.from_proto(self.__expected__.to_model_proto()))
        return result.model

    # Tests applying the __passes__ to the __model__ without relating it to the
    # expected model
    def test_apply_and_verify(self):
        self.passes(self.model)

    # Tests the result of applying the __passes__ to the __model__ to be
    # isomorphic to the __expected__ graph
    def test_isomorphic_to_expected(self):
        _assert_onnx_proto_equal(
            ir.to_proto(self.passes(self.model).model),
            ir.to_proto(self.expected)
        )
