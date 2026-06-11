# Uses pytest for testing, test discovery and reporting
import pytest

from typing import Callable

# Dynamically import python modules at runtime used for dynamically registering
# passes according to configuration files
import importlib

# Passes are done in ONNX IR representation
import onnx_ir as ir

# For comparing constant tensors attached to op inputs
import numpy as np

# Use ONNX Script for creating test models
from onnxscript import OnnxFunction

# Collect custom ONNX IR passes from the library by name
from onnx_passes.passes import collect

# Graph isomorphism utility for ONNX IR models
from onnx_passes.utils.networkx import is_isomorphic


def _format_exception_chain(exc: BaseException) -> str:
    """Return a compact error summary from an exception cause/context chain."""
    lines = []
    current = exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).strip() or repr(current)
        lines.append(f"- {type(current).__name__}: {message}")
        current = current.__cause__ or current.__context__
    return "\n".join(lines)


# Converts ONNX IR objects (or protobuf objects) to deterministic bytes.
def _to_proto_bytes(obj):
    if hasattr(obj, "SerializeToString"):
        return obj.SerializeToString()
    proto = ir.to_proto(obj)
    if hasattr(proto, "SerializeToString"):
        return proto.SerializeToString()
    return repr(proto).encode("utf-8")


# Converts an ONNX IR attribute to a deterministic python value for equality
# comparisons.
def _attr_to_python(attr: ir.Attr):
    if attr.is_ref():
        return ("__ref__", attr.ref_attr_name, int(attr.type))

    t = attr.type
    if t == ir.AttributeType.UNDEFINED:
        return None
    if t == ir.AttributeType.INT:
        return int(attr.as_int())
    if t == ir.AttributeType.FLOAT:
        return float(attr.as_float())
    if t == ir.AttributeType.STRING:
        return str(attr.as_string())
    if t == ir.AttributeType.TENSOR:
        return _to_proto_bytes(attr.as_tensor())
    if t == ir.AttributeType.GRAPH:
        return _to_proto_bytes(attr.as_graph())
    if t == ir.AttributeType.INTS:
        return tuple(int(v) for v in attr.as_ints())
    if t == ir.AttributeType.FLOATS:
        return tuple(float(v) for v in attr.as_floats())
    if t == ir.AttributeType.STRINGS:
        return tuple(str(v) for v in attr.as_strings())
    if t == ir.AttributeType.TENSORS:
        return tuple(_to_proto_bytes(v) for v in attr.as_tensors())
    if t == ir.AttributeType.GRAPHS:
        return tuple(_to_proto_bytes(v) for v in attr.as_graphs())
    if t == ir.AttributeType.SPARSE_TENSOR:
        return _to_proto_bytes(attr.value)
    if t == ir.AttributeType.SPARSE_TENSORS:
        return tuple(_to_proto_bytes(v) for v in attr.value)
    if t == ir.AttributeType.TYPE_PROTO:
        return _to_proto_bytes(attr.value)
    if t == ir.AttributeType.TYPE_PROTOS:
        return tuple(_to_proto_bytes(v) for v in attr.value)

    raise ValueError(f"Unsupported attribute type: {t}")


# Asserts two models are isomorphic and then verifies per-op attributes and
# constant-input values match.
def _assert_onnx_models_equal(model_1: ir.Model, model_2: ir.Model):
    assert is_isomorphic(model_1, model_2)

    nodes_1 = list(ir.traversal.RecursiveGraphIterator(model_1.graph))
    nodes_2 = list(ir.traversal.RecursiveGraphIterator(model_2.graph))

    assert len(nodes_1) == len(nodes_2)

    for n1, n2 in zip(nodes_1, nodes_2):
        assert n1.op_type == n2.op_type
        assert n1.domain == n2.domain
        assert len(n1.inputs) == len(n2.inputs)

        attrs_1 = {k: _attr_to_python(v) for k, v in n1.attributes.items()}
        attrs_2 = {k: _attr_to_python(v) for k, v in n2.attributes.items()}

        assert attrs_1.keys() == attrs_2.keys()
        for key in attrs_1:
            v1, v2 = attrs_1[key], attrs_2[key]
            if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
                assert np.array_equal(v1, v2)
            else:
                assert v1 == v2

        # Compare constant-valued inputs for each op input position.
        for i1, i2 in zip(n1.inputs, n2.inputs):
            if i1 is None or i2 is None:
                assert i1 is None and i2 is None
                continue

            c1 = ir.convenience.get_const_tensor(i1)
            c2 = ir.convenience.get_const_tensor(i2)

            if c1 is None and c2 is None:
                continue

            assert c1 is not None and c2 is not None
            assert np.array_equal(c1.numpy(), c2.numpy())


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
    __model__: OnnxFunction | None = None

    # ONNX Script function for exporting the expected model: Should be a method
    # decorated as @script and @staticmethod
    __expected__: OnnxFunction | None = None

    # Function for generating input arrays for model verification: This is
    # optional, if there is no input generator, verification will be disabled
    __inputs__: Callable | None = None

    # Shared configuration dictionary for all pass instances collected from the
    # __passes__ and __common__ lists
    __config__: dict | None = None

    # Shared state dictionary for all pass instances collected from the
    # __passes__ and __common__ lists - can be used to seed the state
    __state__: dict | None = None

    # Registers a derived test class in the target module/global scope. This
    # keeps generated tests close to the original class-based structure.
    @classmethod
    def register_case(
        cls,
        scope: dict,
        name: str,
        *,
        model: OnnxFunction | None = None,
        expected: OnnxFunction | None = None,
        inputs: Callable | None = None,
        passes: list | None = None,
        common: list | None = None,
        config: dict | None = None,
        state: dict | None = None,
    ):
        attrs = {}

        if model is not None:
            attrs["__model__"] = model
        if expected is not None:
            attrs["__expected__"] = expected
        if inputs is not None:
            attrs["__inputs__"] = staticmethod(inputs)
        if passes is not None:
            attrs["__passes__"] = passes
        if common is not None:
            attrs["__common__"] = common
        if config is not None:
            attrs["__config__"] = config
        if state is not None:
            attrs["__state__"] = state

        # Generated subclasses should be collected by pytest even if the
        # template class uses __test__ = False.
        attrs.setdefault("__test__", True)

        case = type(name, (cls,), attrs)
        case.__module__ = cls.__module__
        scope[name] = case
        return case

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

        state_dict = self.__state__
        if state_dict is None:
            state_dict = {}
            self.__state__ = state_dict

        if self.__inputs__ is not None:
            state_dict["reference"] = self.reference

        return state_dict

    @property
    def config(self):
        __tracebackhide__ = True

        config = self.__config__
        if config is None:
            config = {}
            self.__config__ = config

        # Provide verification reference from eager execution if no explicit
        # reference is configured.
        if self.__inputs__ is not None and "reference" not in config:
            config["reference"] = self.reference

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
        __tracebackhide__ = True

        try:
            self.passes(self.model)
        except Exception as e:
            pytest.fail(
                f"Pass application/verification failed:\n{_format_exception_chain(e)}",
                pytrace=False,
            )

    # Tests the result of applying the __passes__ to the __model__ to be
    # isomorphic to the __expected__ graph
    def test_isomorphic_to_expected(self):
        __tracebackhide__ = True

        try:
            _assert_onnx_models_equal(
                self.passes(self.model).model,
                self.expected,
            )
        except Exception as e:
            pytest.fail(
                f"Model comparison failed:\n{_format_exception_chain(e)}",
                pytrace=False,
            )
