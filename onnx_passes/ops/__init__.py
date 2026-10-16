# ONNX IR values and datatypes
import onnx_ir as ir

# Build a class hierarchy of ONNX operators starting from ABC
from abc import abstractmethod, ABC

# Declaring a custom ONNX opset with name and version
from onnxscript.values import Opset
# Use onnxscript scripts for authoring custom operators as model local functions
from onnxscript import script, OnnxFunction

# Custom operator domain for all operator defined in this package
DOMAIN = "onnx_passes.ops"

# Registry of ONNX operator implementations: Do not use directly, deriving from
# OnnxOperator base class registers implementations to and resolve_operator gets
# implementations from the registry.
_registry = {}


class OnnxOperator(ABC):
    """Base class for ONNX operator implementation and registration."""

    def __init_subclass__(
            cls, domain: str | None = None, version: int | None = None,
            op_type: str | None = None
    ):
        """Registers the subclass implementing the ONNX operator."""

        # If no operator domain is given, use the module implementing the
        # operator subclass
        if domain is None:
            domain = cls.__module__

        # If not op-type is given explicitly, derive from the class name by
        # stripping away a potential version suffix _vN
        if op_type is None:
            op_type, *_ = cls.__name__.split("_v")

        # If no version is given explicitly, derive from the class name by
        # splitting on the version suffix _v
        if version is None:
            op_type, version = cls.__name__.split("_v")

        # Put the class into the registry uniquely identified by type - domain -
        # version combination
        _registry[(op_type, domain, int(version))] = cls

        # Identify the operator by its op-type and opset (domain+version)
        cls.op_type = op_type
        cls.opset = Opset(domain, int(version))

    def __init__(self, default_opset: Opset):
        """Instantiates the operator with a default opset to use."""
        self.default_opset = default_opset

        # Check whether the operator requires a minimum or maximum version of
        # the default opset raise an exception otherwise
        min_version = -float("inf")
        max_version = +float("inf")

        try:
            min_version = self.MIN_DEFAULT_OPSET_VERSION  # noqa: Maybe subclass
            max_version = self.MAX_DEFAULT_OPSET_VERSION  # noqa: Maybe subclass
        except AttributeError:
            pass

        if not min_version <= default_opset.version <= max_version:
            raise ValueError(
                f"{self.op_type}_v{self.opset.version} requires a default"
                f" opset version between {min_version} and {max_version},"
                f" given: {default_opset.version}"
            )

    def to_function_proto(self):
        """Converts the operator to an ONNX function proto."""
        return self.onnx_function.to_function_proto()

    def to_model_proto(self):
        """Converts the operator to an ONNX model proto."""
        return self.onnx_function.to_model_proto()

    def __call__(self, *args, **kwargs):
        """Eager mode function call evaluation."""
        return self.onnx_function(*args, **kwargs)

    @property
    def onnx_function(self) -> OnnxFunction:
        """Operator ONNX Script wrapping the subclass implementation."""
        # Turn the script implemented by the subclass into an ONNX function in
        # the opset using the default_opset
        f = self.script(self.default_opset)
        f = script(self.opset, self.default_opset)(f)

        # Swap out the traced function name for the operator type used to
        # register this function
        f._name = self.op_type
        f.__name__ = self.op_type
        f.function_ir.name = self.op_type

        return f

    @staticmethod
    @abstractmethod
    def script(op: Opset):
        """Operator ONNX Script - must be implemented by subclass."""
        ...


# Use importlib for dynamic imports
from importlib import import_module


def resolve_op(op_type: str, domain: str = "", version: int | None = None):
    """Resolves the operator implementation from (op-type, domain, version)."""

    # Try directly looking up the operator implementation in the registry first
    # before falling back to dynamic imports and version search.
    try:
        return _registry[(op_type, domain, version)]
    except KeyError:
        # Try dynamically importing (a) relative to the onnx_passes.ops
        # and (b) interpreting the domain as a python module.
        try:
            import_module(f"onnx_passes.ops.{domain}".strip("."))
        except ModuleNotFoundError:
            try:
                import_module(f"{domain}")
            except ModuleNotFoundError:
                pass

        # If no version is specified, try with the largest supported version
        # available in the registry
        if version is None:
            version = max(v for _, _, v in _registry.keys())

        # Resolve the domain and check whether the operator version is within
        # the supported version range (this check is optional)
        if (opset := resolve_domain(domain)) is not None:
            try:
                min_version = opset.MIN_OPSET_VERSION
                max_version = opset.MAX_OPSET_VERSION
                if not min_version <= version <= max_version:
                    raise ValueError(
                        f"Unsupported version {version} for domain '{domain}':"
                        f" must be between {min_version} and {max_version}"
                    )
            except AttributeError:
                pass

        # Now try directly looking up the operator implementation again, while
        # progressively lowering the version until a match is found
        for v in range(version, -1, -1):
            try:
                return _registry[(op_type, domain, v)]
            except KeyError:
                pass

    # Still not found - give up...
    raise KeyError((op_type, domain, version))


def resolve_domain(domain: str):
    """Resolves the module implementing the domain."""

    # Try dynamically importing (a) relative to the onnx_passes.ops
    # and (b) interpreting the domain as a python module.
    try:
        return import_module(f"onnx_passes.ops.{domain}".strip("."))
    except ModuleNotFoundError:
        try:
            return import_module(f"{domain}")
        except ModuleNotFoundError:
            return None


from typing import Sequence

INTS = Sequence[int]


class Im2Col_v1(OnnxOperator):
    """Sliding window generator Im2Col operator."""

    @staticmethod
    def script(op: Opset):
        """Generate an ONNX Script function implementing Im2Col."""

        def im2col(x, kernel_shape: INTS, dilations: INTS, strides: INTS):
            # Im2Col operates in channels last layout, i.e., assumes the input x
            # to be arranged as N x D1 x D2 x ... x Dn x C, where Di are spatial
            # dimensions corresponding to k1 x k2 x ... x kn kernel dimensions.
            #
            # Im2Col leverages a Conv operator internally to gather pixels via a
            # one-hot encoded set of filters. As Conv operates in channels first
            # layout, shuffle the input to N x C x D1 x D2 x ... x Dn.
            x = op.Reshape(
                op.Transpose(
                    op.Reshape(
                        x,
                        op.Concat(
                            op.Shape(x, start=0, end=1),
                            op.Constant(value_ints=[-1]),
                            op.Shape(x, start=-1),
                            axis=0
                        )
                    ),
                    perm=[0, 2, 1]
                ),
                op.Concat(
                    op.Shape(x, start=0, end=1),  # N
                    op.Shape(x, start=-1),  # C
                    op.Shape(x, start=1, end=-1),  # D1 x D2 x ... Dn
                    axis=0
                )
            )

            # Kernel shape and number of input channels as ONNX constant (or at
            # least constant-foldable) nodes.
            k = op.Constant(value_ints=kernel_shape)
            c = op.Shape(x, start=1, end=2)

            # Represent each position in the k1 x k2 x ... x kn kernel via a
            # one-hot encoding: c * Prod{k} x 1 x k1 x k2 x ... x kn
            #
            # Applying this as a convolution filter gathers all pixels for each
            # positioning of the kernel, i.e., this extracts sliding windows of
            # the kernel shape from one C-plane of the input.
            #
            # The general idea of using the one-hot encoding kernel with group
            # convolution is based on
            #       https://github.com/f-dangel/unfoldNd/#the-trick
            gather_kernel = op.Reshape(
                op.Reshape(
                    op.OneHot(
                        op.Expand(
                            op.Range(
                                op.Constant(value_int=0),
                                op.ReduceProd(k),
                                op.Constant(value_int=1),
                            ),
                            op.Concat(
                                c,
                                op.Constant(value_ints=[1]),
                                axis=0
                            )
                        ),
                        op.ReduceProd(
                            k, keepdims=0
                        ),
                        op.Constant(value_ints=[0, 1])
                    ),
                    op.Concat(
                        c,
                        k,
                        op.Constant(value_ints=[1]),
                        k,
                        axis=0
                    )
                ),
                # Two-step Reshape to the desired kernel shape, otherwise shape
                # inference chokes on propagating these shapes to the output...
                op.Concat(
                    op.Constant(value_ints=[-1, 1]),
                    k,
                    axis=0
                )
            )

            # All C-planes share kernel index pattern: shuffle the channel axis
            # into the batch dimension to avoid doing a grouped convolution with
            # C groups. For a proper grouping we would need to (a) replicate the
            # gather kernel C times and (b) lift the number of channels from the
            # shape extracted from the input to the attribute scope (this cannot
            # be expressed in ONNX).
            y = op.Conv(
                op.Reshape(
                    x,
                    op.Concat(
                        op.Constant(value_ints=[-1, 1]),
                        op.Shape(x, start=2),
                        axis=0
                    )
                ),
                op.CastLike(
                    gather_kernel,
                    x
                ),
                kernel_shape=kernel_shape,
                strides=strides,
                dilations=dilations
            )

            y = op.Reshape(
                op.Slice(
                    y,
                    op.Constant(value_ints=[0]),
                    op.ReduceProd(k),
                    op.Constant(value_ints=[1]),
                    op.Constant(value_ints=[1]),
                ),
                op.Concat(
                    op.Shape(x, end=1),  # N
                    op.Shape(gather_kernel, end=1),  # C * Prod{k}
                    op.Shape(y, start=2),  # D1' x D2' x ... Dn'
                    axis=0
                )
            )

            # Shuffle each window, i.e., the inner c * Prod{k} dimensions, into
            # channel-last layout. These dimensions are gathered in flat layout
            # and must be unflattened, transposed and flattened again.
            y = op.Reshape(
                op.Transpose(
                    op.Reshape(
                        y,
                        op.Concat(
                            op.Shape(y, end=1),
                            c,
                            op.ReduceProd(k),
                            op.Constant(value_ints=[-1]),
                            axis=0
                        )
                    ),
                    perm=[0, 2, 1, 3]
                ),
                op.Concat(
                    op.Shape(y, end=1),
                    op.Shape(gather_kernel, end=1),
                    op.Shape(y, start=2),
                    axis=0
                )
            )

            # Shuffle the output back to channels-last layout such that from the
            # outside Im2Col appears to operate in channels-last.
            return op.Reshape(
                op.Transpose(
                    op.Reshape(
                        y,
                        op.Concat(
                            op.Shape(y, start=0, end=1),
                            op.Shape(y, start=1, end=2),
                            op.Constant(value_ints=[-1]),
                            axis=0
                        )
                    ),
                    perm=[0, 2, 1]
                ),
                op.Concat(
                    op.Shape(y, start=0, end=1),  # N
                    op.Shape(y, start=2),  # D1' x D2' x ... Dn'
                    op.Shape(y, start=1, end=2),  # C * Prod{k1 x k2 x ... x kn}
                    axis=0
                )
            )

        return im2col


# ONNX list attributes must be annotated as Sequence in ONNX Script
from typing import Sequence


class LayoutConverter_v1(OnnxOperator):
    """Converts between data layouts.

    Syntactically this acts just as a transpose but allows to attach some
    semantics via the assumes attribute (which otherwise is ignored by the
    operator).

    The LayoutConverter itself does not interact with the usual streamlining
    flow and can thus be used to demarcate sections of the graph operating on
    different data layouts.

    Intended usage is for switching between channels-first and channels-last
    layout of image-like data, or as a marker for custom checks or
    graph-surgery transformations.
    """

    @staticmethod
    def script(op: Opset):
        """Generate an ONNX Script function implementing LayoutConverter."""

        def layout_converter(x, assumes: Sequence[str], perm: Sequence[int]):
            return op.Transpose(x, perm=perm)

        return layout_converter


class Round_v1(OnnxOperator):
    """Custom rounding function with configurable rounding mode via attribute
    matching QONNX/Brevitas behavior.

    Primarily meant to be used inside the Quant operator implementation.
    """

    # Implements configurable rounding mode via string comparison inside the
    # graph, Equal supports string comparison since opset 19
    MIN_DEFAULT_OPSET_VERSION: int = 19

    @staticmethod
    def script(op: Opset):
        """Generate an ONNX Script function implementing Quant."""

        def round(x, rounding_mode: str = "ROUND"):  # noqa: Shadows built-in
            # Predefined rounding mode constants for readability...
            ROUND = op.Constant(value_string="ROUND")
            CEIL = op.Constant(value_string="CEIL")
            FLOOR = op.Constant(value_string="FLOOR")
            ROUND_TO_ZERO = op.Constant(value_string="ROUND_TO_ZERO")

            # It is not possible to return from within If-Else branches, assign
            # to a temporary within each branch
            if rounding_mode == ROUND:
                y = op.Round(x)
            elif rounding_mode == CEIL:
                y = op.Ceil(x)
            elif rounding_mode == FLOOR:
                y = op.Floor(x)
            elif rounding_mode == ROUND_TO_ZERO:
                y = op.Mul(op.Sign(x), op.Floor(op.Abs(x)))
            else:
                # Else branch cannot be omitted, and it is not possible to raise
                # exceptions or assertions - fallback to Round...
                y = op.Round(x)

            # Return output from branch selected by rounding mode
            return y

        return round


class Quant_v1(OnnxOperator):
    """QONNX quantizer custom operator implementation to allow models with
    custom quantization to be executed via ONNX Runtime.

    See https://github.com/fastmachinelearning/qonnx for details...
    """

    @staticmethod
    def script(op: Opset):
        """Generate an ONNX Script function implementing Quant."""

        round = Round_v1(op).onnx_function  # noqa: Shadows built-in

        def quant(x, scale, zeropoint, bitwidth, signed: int, narrow: int,
                  rounding_mode: str):
            # Quantizer attributes are specified as integers but are use in
            # calculations together with float inputs - inputs to Add, Mul, etc.
            # must match in type
            signed = op.Cast(signed, to=ir.DataType.FLOAT)
            narrow = op.Cast(narrow, to=ir.DataType.FLOAT)

            # Minimum representable integer of signed bitwidth taking narrow
            # range into account - calculations inlined into the graph, depends
            # on dynamic bitwidth
            _min = (- 2.0 ** (bitwidth - signed) + narrow) * signed

            # Maximum representable integer of signed bitwidth taking narrow
            # range into account - calculations inlined into the graph, depends
            # on dynamic bitwidth
            _max = 2.0 ** (bitwidth - signed) - 1 - narrow * (1 - signed)

            # Scale and zero point: Float to Integer
            q = op.Add(op.Div(x, scale), zeropoint)

            # This simulates if-else branching without an if operator - usually
            # the condition should eventually evaluate to a constant expression
            # allowing one branch to be eliminated. op.Where also takes care of
            # broadcasting.
            q = op.Where(
                # Condition: if bitwidth == 1 and signed - signed 1-bit needs
                # manual fix...
                op.And(
                    op.Equal(bitwidth, 1.0),
                    op.Cast(signed, to=ir.DataType.BOOL)
                ),
                # If-branch: Fix 1-bit quantization as manually converted
                # bipolar encoding
                op.Where(
                    op.GreaterOrEqual(q, 0.0), op.CastLike(1.0, q),
                    op.CastLike(-1.0, q)
                ),
                # Else-branch: Clip the integer to the range and round according
                # to the rounding mode while ensuring the data type to stay the
                # same
                round(op.Clip(q, _min, _max), rounding_mode=rounding_mode)
            )

            # Scale and zero point: Integer to Float
            return op.Mul(op.Sub(q, zeropoint), scale)

        return quant


class MultiThreshold_v1(OnnxOperator):
    """Multi-Threshold operator for quantized activations and layer tails.
    """

    @staticmethod
    def script(op: Opset):
        """Generate an ONNX Script function implementing MultiThreshold."""

        def multithreshold(x, thresholds, weights):
            # Comparison of inputs and all corresponding thresholds: Expand
            # input dimensions to match the threshold parameter shape via
            # broadcasting
            steps = op.GreaterOrEqual(op.Unsqueeze(x, axes=[-1]), thresholds)
            # Type-casting turns boolean unit steps to reducible floats
            steps = op.Cast(steps, to=ir.DataType.FLOAT)
            # Finally the multi-threshold output reduces over all steps removing
            # the previously expanded dimension
            return op.ReduceSum(weights * steps, [-1], keepdims=0)

        return multithreshold


class Any_v1(OnnxOperator):
    """Yields True if any value in input is True (or non-zero)."""

    @staticmethod
    def script(op: Opset):
        """Generate an ONNX Script function implementing Any."""

        def any(x):  # noqa: Shadows built-in
            return op.Greater(
                op.ReduceMax(
                    op.Abs(op.Cast(x, to=ir.DataType.INT64)), keepdims=0
                ),
                0
            )

        return any


class Log2_v1(OnnxOperator):
    """Logarithm base 2."""

    @staticmethod
    def script(op: Opset):
        """Generate an ONNX Script function implementing Log2."""

        def log2(x):
            return op.Where(
                op.Equal(
                    x,
                    op.Constant(value_float=0.0)
                ),
                op.Constant(value_float=-float("inf")),
                op.Div(
                    op.Log(
                        op.Where(
                            op.Equal(
                                x,
                                op.Constant(value_float=0.0)
                            ),
                            op.Add(
                                x,
                                op.Constant(value_float=1.0)
                            ),
                            x
                        )
                    ),
                    op.Log(op.CastLike(op.Constant(value_float=2.0), x))
                )
            )

        return log2


class Ulp_v1(OnnxOperator):
    """Evaluates the unit in the last place (ULP) of floating point inputs x.

    The ULP is the spacing between two consecutive floating-point numbers.
    """

    @staticmethod
    def script(op: Opset):
        """Generate an ONNX Script function implementing Ulp."""

        log2 = Log2_v1(op).onnx_function

        def ulp(x):
            return op.Where(
                op.IsNaN(x),
                x,
                op.Where(
                    op.IsInf(x),
                    op.Abs(x),
                    op.Pow(
                        op.Constant(value_float=2.0),
                        op.Sub(
                            op.Max(
                                op.Floor(log2(op.Abs(x))),
                                op.Constant(value_float=-126),
                            ),
                            op.Constant(value_float=23)
                        )
                    )
                )
            )

        return ulp


def link_ops_from_graph(model: ir.Model, graph: ir.Graph) -> ir.Model:
    """Links functions implementing custom-Ops into the ONNX model."""

    # First pass over the nodes: Track the maximum version used for all opsets
    # which provide operators we can resolve, i.e., are responsible for
    opset_imports = {}

    for node in ir.traversal.RecursiveGraphIterator(graph):
        # Resolve the node opset version either from the node or falling back to
        # graph opset imports
        if (version := node.version) is None:
            version = model.graph.opset_imports[node.domain]

        # Look up the operator from the registry and skip if there is no special
        # function to be linked for this operator
        try:
            operator = resolve_op(node.op_type, node.domain, version)
        except KeyError:
            continue

        if node.domain not in opset_imports:
            opset_imports[node.domain] = operator.opset.version

        # Track the maximum of the already imported and actually resolved
        # version
        opset_imports[node.domain] = max(
            operator.opset.version, opset_imports[node.domain]
        )

    for opset, version in opset_imports.items():
        if opset not in model.graph.opset_imports:
            model.graph.opset_imports[opset] = version

    # Second pass over the nodes: Resolve all operators and check their validity
    # with respect to node and graph opset import version
    for node in ir.traversal.RecursiveGraphIterator(graph):
        # Resolve the node opset version either from the node or falling back to
        # graph opset imports
        if (version := node.version) is None:
            version = model.graph.opset_imports[node.domain]

        # Look up the operator from the registry and skip if there is no special
        # function to be linked for this operator
        try:
            operator = resolve_op(node.op_type, node.domain, version)
        except KeyError:
            if node.op_identifier() in model.functions:
                link_ops_from_graph(
                    model, model.functions[node.op_identifier()].graph
                )

            continue

        # If resolving the operator from the opset import yields a different
        # version, this seems to be a case of non-standard mixed-opset versions
        import_version = model.graph.opset_imports[node.domain]
        if resolve_op(node.op_type, node.domain,
                      import_version).opset.version != operator.opset.version:
            raise SyntaxError(
                f"Mixed opset not supported for {node.domain} v{version}:"
                f" already imported v{model.graph.opset_imports[node.domain]}"
            )

        # Instantiate the operator with the version of the default opset im
        # ported by the model graph and convert to ONNX IR
        operator = operator(Opset("", model.graph.opset_imports[""]))
        operator_ir: ir.Function = ir.from_proto(operator.to_function_proto())

        # Recursively link functions from the function graphs into the model to
        # discover functions used indirectly by subgraphs
        if operator_ir.identifier() not in model.functions:
            link_ops_from_graph(model, operator_ir.graph)

            # Insert the ONNX IR function into the function list of the model
            model.functions[operator_ir.identifier()] = operator_ir

    return model


def link_ops(model: ir.Model) -> ir.Model:
    """Links functions implementing custom-Ops into the ONNX model."""

    if model.graph is None:
        return model

    link_ops_from_graph(model, model.graph)

    return model
