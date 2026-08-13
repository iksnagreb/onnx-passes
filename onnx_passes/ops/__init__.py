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


class ArgSort_v1(OnnxOperator):
    """Sort values by argument along an axis."""

    @staticmethod
    def script(op: Opset):
        """Generate an ONNX Script function implementing ArgSort."""

        def argsort(x, axis: int = -1):
            # The following abuses the TopK operation to sort all elements along
            # the axis. We set k to the total number of elements along axis.
            k = op.GatherElements(
                op.Shape(x),
                op.Expand(
                    op.Constant(value_int=axis),
                    op.Constant(value_ints=[1])
                )
            )

            # TopK sorts the input into ascending order (largest=0) along axis
            # by setting k = x.shape[axis], we only care for the indices...
            _, indices = op.TopK(x, k, axis=axis, largest=0)

            return indices

        return argsort


class Im2Col_v1(OnnxOperator):
    """Sliding window generator Im2Col operator."""

    @staticmethod
    def script(op: Opset):
        """Generate an ONNX Script function implementing Im2Col."""

        def im2col(x, indices):
            return op.Gather(op.Flatten(x), indices, axis=1)

        return im2col


class Swish_v1(OnnxOperator):
    """Swish function x * Sigmoid(alpha * x)."""

    @staticmethod
    def script(op: Opset):
        """Generate an ONNX Script function implementing Swish."""

        def swish(x, alpha: float = 1.0):
            return x * op.Sigmoid(alpha * x)

        return swish


class Silu_v1(OnnxOperator):
    """Silu function x * Sigmoid(x)."""

    @staticmethod
    def script(op: Opset):
        """Generate an ONNX Script function implementing Silu."""

        def silu(x):
            return x * op.Sigmoid(x)

        return silu


class LambertW_v1(OnnxOperator):
    """Lambert W function.

    Approximation via recursive formula according to R. Iacono and J.P. Boyd
    2017, with starting values according to Lóczi, Lajos 2022.

    Note: Only the real-valued primary (k=0) and secondary branch (k=-1) are
    implemented. For any other value of k this yields NaN.
    """

    @staticmethod
    def script(op: Opset):
        """Generate an ONNX Script function implementing LambertW."""

        def lambertw(
                x, k: int = 0, iterations: int = 4, tolerance: float = 1.0e-8
        ):
            # Define Euler's number as a constant node as it is used frequently
            # down below
            e = op.Exp(op.Constant(value_float=1.0))

            # Select starting values depending on k and x according to Lóczi,
            # Lajos 2022
            w = op.Where(
                # Principal branch of Lambert W with -1/e < x < infinity
                k == 0,
                op.Where(
                    e < x,
                    op.Log(x) - op.Log(op.Log(x)),
                    op.Where(
                        op.Constant(value_float=0.0) <= x,
                        op.Div(x, e),
                        op.Div(
                            e * x * op.Log(1 + op.Sqrt(1 + e * x)),
                            1 + e * x + op.Sqrt(1 + e * x)
                        )
                    )
                ),
                op.Where(
                    # Secondary branch of Lambert W with -1/e < x < 0
                    k == -1,
                    op.Where(
                        op.Constant(value_float=-0.25) < x,
                        op.Log(op.Neg(x)) - op.Log(op.Neg(op.Log(op.Neg(x)))),
                        -1 - op.Sqrt(2.0) * op.Sqrt(1 + e * x)
                    ),
                    # Fallback: As ONNX Script needs to trace all branches and
                    # cannot raise, NaN is returned for k not in {0,-1}
                    op.Constant(value_float=float("NaN"))
                )
            )

            # Quadratic-rate recursive formula according to R. Iacono and J.P.
            # Boyd 2017
            for i in range(iterations):
                w = (w / (1 + w)) * (1 + op.Log(op.Div(x, w)))

            # Approximation of Lambert W_{k}(x), insert the exact results for
            # values close to the branch point -1/e to avoid numerical issues
            return op.Where(
                op.Abs(op.Add(x, op.Reciprocal(e))) <= tolerance, -1, w
            )

        return lambertw


class InverseSwish_v1(OnnxOperator):
    """Inverse of the Swish function.

    As Swish is not invertible, this yields branches expressed in terms of the
    Lambert W function for inputs for which a real-valued solution of the
    inverse exists.

    All other inputs are mapped to the appropriate -/+ infinity, such that the
    inverse behaves nicely with respect to comparison (>=)/thresholding.

    Note: Only the real-valued primary (k=0) and secondary branch (k=-1) are
    implemented. For any other value of k this yields NaN.
    """

    @staticmethod
    def script(op: Opset):
        """Generate an ONNX Script function implementing InverseSwish."""

        swish = Swish_v1(op).onnx_function
        lambertw = LambertW_v1(op).onnx_function

        def inverse_swish(
                x, alpha: float = 1.0, k: int = 0, tolerance: float = 1.0e-8
        ):
            # Value at the global minimum of the Swish function is the smallest
            # input for which a real-valued solution of the inverse exists
            x_min = swish(
                -alpha ** -1 * (1.0 + lambertw(op.Exp(-1.0))), alpha=alpha
            )

            # Short aliases to infinity and NaN used for out of range inputs
            inf = op.Constant(value_float=float("inf"))
            nan = op.Constant(value_float=float("nan"))

            # Evaluate the Lambert W function on the selected branch: For inputs
            # with real-valued solutions, the inverse is given as
            #   Swish^{-1}_{k}(x)
            #       = x + alpha**-1 * W_{k}(alpha * x * e**(-alpha * x))
            w = lambertw(alpha * x * op.Exp(-alpha * x), k, tolerance=tolerance)

            # Select from principal (k=0) and secondary (k=-1) branch or fall
            # back returning NaN for unsupported branches
            return op.Where(
                k == 0,
                op.Where(
                    x >= x_min, op.Where(x >= inf, inf, x + alpha ** -1 * w),
                    -inf
                ),
                op.Where(
                    k == -1,
                    op.Where(
                        x < 0.0, op.Where(x >= x_min, x + alpha ** -1 * w, inf),
                        -inf
                    ),
                    # Fallback: As ONNX Script needs to trace all branches and
                    # cannot raise, NaN is returned for k not in {0,-1}
                    nan
                )
            )

        return inverse_swish


class InverseSilu_v1(OnnxOperator):
    """Inverse of the Silu function.

    As Silu is not invertible, this yields branches expressed in terms of the
    Lambert W function for inputs for which a real-valued solution of the
    inverse exists.

    Note: As Silu is defined as the Swish functions with alpha=1.0, we can
    define the inverse of Silu as a special case of the inverse Swish.
    """

    @staticmethod
    def script(op: Opset):
        """Generate an ONNX Script function implementing InverseSilu."""

        inverse_swish = InverseSwish_v1(op).onnx_function

        def inverse_silu(x, k: int = 0, tolerance: float = 1.0e-8):
            return inverse_swish(x, alpha=1.0, k=k, tolerance=tolerance)

        return inverse_silu


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
            # Type-casing turns boolean unit steps to reducible floats
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
                ), 0
            )

        return any


class Log2_v1(OnnxOperator):
    """Logarithm base 2."""

    @staticmethod
    def script(op: Opset):
        """Generate an ONNX Script function implementing Log2."""

        def log2(x):
            return op.Log(x) / op.Log(
                op.CastLike(op.Constant(value_float=2.0), x)
            )

        return log2


class Ulp_v1(OnnxOperator):
    """Evaluates the unit in the last place (ULP) of floating point inputs x.

    The ULP is the spacing between two consecutive floating-point numbers.
    """

    @staticmethod
    def script(op: Opset):
        """Generate an ONNX Script function implementing Ulp."""

        any = Any_v1(op).onnx_function  # noqa: Shadows built-in
        log2 = Log2_v1(op).onnx_function

        def ulp(x):
            # Define basic constants matching the type of the input to keep the
            # following more terse and readable
            _0 = op.CastLike(op.Constant(value_float=0.0), x)
            _1 = op.CastLike(op.Constant(value_float=1.0), x)
            _2 = op.CastLike(op.Constant(value_float=2.0), x)

            # Get rid of infinities...
            infinity, x = op.IsInf(x), op.Where(op.IsInf(x), _1, x)

            # Round the input down to the nearest power of two and sanitize zero
            # inputs to avoid taking the log of zero
            x = op.Where(
                op.Not(infinity),
                op.Where(
                    x == _0,
                    x,
                    op.Pow(
                        _2,
                        op.Floor(log2(op.Abs(x) + op.Where(x == _0, _1, _0)))
                    )
                ),
                x
            )

            # Start searching for the exponent of the Ulp(x) in the middle of
            # the range, expanding to the full shape of the input, as we want
            # the ulp per element
            exp = op.Expand(_0, op.Shape(x))

            # Increase the ulp exponent while x + ulp == x, this covers all x
            # for which the Ulp(x) is >=1
            condition = any(
                op.And(x + op.Pow(_2, exp) == x, op.Not(op.IsInf(x)))
            )
            while condition:
                exp = exp + op.Where(x + op.Pow(_2, exp) == x, _1, _0)
                condition = any(
                    op.And(x + op.Pow(_2, exp) == x, op.Not(op.IsInf(x)))
                )

            # As the stop condition stops at the exponent where no difference is
            # observed, take back one step to get the Ulp(x)
            exp = exp - _1

            # Decrease the ulp exponent while x + ulp > x, this covers all x for
            # which the Ulp(x) is <=1
            condition = any(
                op.And(x + op.Pow(_2, exp) > x, op.Not(op.IsInf(x)))
            )
            while condition:
                exp = exp - op.Where(x + op.Pow(_2, exp) > x, _1, _0)
                condition = any(
                    op.And(x + op.Pow(_2, exp) > x, op.Not(op.IsInf(x)))
                )

            # As the stop condition stops at the exponent where no difference is
            # observed, add back one step to get the Ulp(x)
            return op.Where(infinity, _0, op.Pow(_2, exp + _1))

        return ulp


class PowerQuantMatMul_v1(OnnxOperator):
    """Matrix multiplication resulting from PowerQuant.

    See: https://arxiv.org/abs/2301.09858
    """

    @staticmethod
    def script(op: Opset):
        """Generate an ONNX Script function implementing PowerQuantMatMul."""

        def power_quant_matmul(x, y, alpha):
            return op.Mul(
                op.MatMul(
                    op.Round(
                        op.Mul(
                            op.Mul(
                                op.Sign(x),
                                op.Pow(
                                    op.Abs(x),
                                    op.Reciprocal(alpha)
                                )
                            ),
                            op.Constant(value_float=2 ** 23)
                        )
                    ),
                    op.Round(
                        op.Mul(
                            op.Mul(
                                op.Sign(y),
                                op.Pow(
                                    op.Abs(y),
                                    op.Reciprocal(alpha)
                                )
                            ),
                            op.Constant(value_float=2 ** 23)
                        )
                    )
                ),
                op.Constant(value_float=2 ** -23)
            )

        return power_quant_matmul


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


# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# Inserting custom ops is considered as an annotation pass as it does not really
# modify the model graph structure or values
from onnx_passes.passes.base import Pass, InPlacePass


# Annotation pass inserting custom operator functions into the model
@passes.register("link-ops")
class LinkCustomOps(Pass, InPlacePass):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return ir.passes.PassResult(link_ops(model), False)
