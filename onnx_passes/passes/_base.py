# Path manipulation and directory creation: os.path.join, os.makedirs
import os
# Inspect python object, e.g., to test whether they are a module
import inspect

# Build a class hierarchy of ONNX passes starting from ABC
from abc import ABC, abstractmethod
# Path to files or directories
from pathlib import Path
# Type hints for annotating the interface
from typing import Any, Callable

# ONNX intermediate representation
import onnx_ir as ir

# Pass configuration and state tracking structures
from onnx_passes.passes._config import Config
from onnx_passes.passes._state import State


# Exception indicating verification failure evaluating the model - currently
# does not do add anything ontop the base Exception.
class VerificationError(Exception):
    ...


# Registry of ONNX pass implementations: Do not use directly, deriving from Pass
# base class registers implementations to and resolve_pass gets implementations
# from the registry.
_registry = {}


class Pass(ir.passes.PassBase, ABC):
    """Base class for ONNX passes implementation and registration.

    The base class takes care of registering new passes and uniformly
    orchestrating pass application, verification and logging.
    """

    def __init_subclass__(
            cls, module: str | None = None, name: str | None = None,
            version: int | None = None
    ):
        """Registers the subclass implementing the ONNX pass."""

        # Only register fully specialized classes, partial specializations are
        # marked abstract by deriving from ABC once more
        if ABC not in cls.__bases__:
            # If no pass module is given, use the module implementing the
            # operator subclass
            if module is None:
                module = cls.__module__

            # If no version is given explicitly, derive from the class name by
            # splitting on the version suffix _v
            if version is None:
                name, version = cls.__name__.split("_v")

            # Put into the registry uniquely identified by module-name-version
            # combination
            _registry[(module, name, int(version))] = cls

    @property
    def identifier(self):
        return f"{self.__class__.__module__}.{self.__class__.__name__}"

    def __init__(self, config: Config = Config()):
        """Initializes the pass with a pass configuration."""
        self.config = config

    def __call__(self, model_or_result: ir.Model | ir.passes.PassResult, /):
        """Applies the pass to the model or result from previous pass."""

        # Accepts both, ir.Model and ir.passes.PassResult from any previous pass
        # application in a sequence of passes
        if isinstance(model_or_result, ir.passes.PassResult):
            model = model_or_result.model
        else:
            model = model_or_result

        # Write to log when entering the pass: Always log the pass to the
        # metadata and if configured write to the output and store checkpoints
        if self.config.logging.verbose:
            print(f"Entering {self.identifier}")

        if self.config.logging.checkpoint:
            ir.save(model, f"before-{self.config.logging.checkpoint}")

        model.meta.setdefault("passes", State()).log_pass(self)

        # Evaluate the precondition on the model, which raises an exception to
        # indicate failure. Pass through any PreconditionError and wrap all
        # other exceptions in PreconditionError as well
        try:
            self.requires(model)
        except ir.passes.PreconditionError:
            raise
        except Exception as e:
            raise ir.passes.PreconditionError(
                f"Pre-condition for pass '{self.__class__.__name__}' failed"
            ) from e

        # Prepares the model and state for verification, e.g., by computing
        # expected outputs with the not-yet transformed model
        try:
            self.prepare_verify(model)
        except VerificationError:
            raise
        except Exception as e:
            raise VerificationError(
                f"Verification of pass '{self.__class__.__name__}' failed"
            ) from e

        # Call the pass implementation (provided by the specialization) on the
        # model producing a PassResult
        result = self.cleanup(self.call(model))

        # Ensure the implementation respects the API signature and yields a
        # PassResult and not simply a model or something entirely different
        if not isinstance(result, ir.passes.PassResult):
            raise TypeError(
                f"The result of the pass '{self.__class__.__name__}' should be"
                f" type PassResult."
                f" Please create one with ir.passes.PassResult()."
            )

        # Ensure the implementation respects the declared properties/categories
        # regarding in-place pass application
        if self.in_place and result.model is not model:
            raise ir.passes.PassError(
                f"The pass '{self.__class__.__name__}' is declared in-place,"
                f" but the model returned is *not* the same object as the input"
                f" model. Pass developer: Pass should return the same model"
                f" object or the in_place property should return False."
            )

        if not self.in_place and result.model is model:
            raise ir.passes.PassError(
                f"The pass '{self.__class__.__name__}' is declared not"
                f" in-place, but the model returned *is* the same object as the"
                f" input model. Pass developer: Pass should return a new model"
                f" object or the in_place property should return True."
            )

        # Write to log when leaving the pass: Always log the pass to the
        # metadata and if configured write to the output and store checkpoints
        if self.config.logging.checkpoint:
            ir.save(result.model, f"after-{self.config.logging.checkpoint}")

        # Detailed logging of intermediate models after pass application
        if self.config.logging.keep_intermediates:
            # Get the logging directory pathname
            path = Path(self.config.logging.keep_intermediates)  # noqa: Path
            # Make sure the directory exists...
            os.makedirs(path, exist_ok=True)
            # Mark this as the after-the-pass checkpoint
            filename = os.path.join(path, f"{model.meta['passes'].id}.onnx")
            # Save the model checkpoint
            ir.save(result.model, filename)

        # Evaluate the postcondition on the pass result (model and indication on
        # whether the model has been modified), which raises an exception to
        # indicate failure. Pass through any PostconditionError and wrap all
        # other exceptions in PostconditionError as well
        try:
            self.ensures(result)
        except ir.passes.PostconditionError:
            raise
        except Exception as e:
            raise ir.passes.PostconditionError(
                f"Post-condition for pass '{self.__class__.__name__}' failed"
            ) from e

        # Finish verification of the transformed model, e.g., by comparing to
        # expected outputs from the not transformed model
        try:
            self.verify(result)
        except VerificationError:
            raise
        except Exception as e:
            raise VerificationError(
                f"Verification of pass '{self.__class__.__name__}' failed"
            ) from e

        # Final log message when leaving the pass
        if self.config.logging.verbose:
            print(f"Leaving {self.identifier}")

        return result

    def requires(self, model: ir.Model):
        """Evaluates the precondition on the model."""
        ...

    def ensures(self, result: ir.passes.PassResult):
        """Evaluates the postcondition on the pass result."""
        ...

    def prepare_verify(self, model: ir.Model):
        """Prepare model verification before applying the pass."""
        ...

    def verify(self, result: ir.passes.PassResult):
        """Verify the model after applying the pass."""
        ...

    @staticmethod
    def cleanup(result: ir.passes.PassResult):
        """Performs cleanup of the pass result."""
        return result


# Use importlib for dynamic imports
from importlib import import_module


def resolve_passes(identifier: Any):
    """Resolves a passes from the identifier string."""

    # Pass through already resolved pass classes and resolve module instances by
    # using the module name as the identifier
    if isinstance(identifier, type) and issubclass(identifier, Pass):
        return [identifier]

    if inspect.ismodule(identifier):
        identifier: str = identifier.__name__

    if not isinstance(identifier, str):
        raise SyntaxError(
            f"Illegal pass identifier: {identifier}"  # noqa: Might not have str
        )

    # First try to interpret as a fully qualified pass identifier, i.e., a
    # complete <module>.<name>_v<version> string
    *module, cls = identifier.split(".")
    name, *version = cls.split("_v")

    module = ".".join(module)

    if len(version) > 1:
        raise SyntaxError(f"Illegal pass identifier: {identifier}")

    try:
        version = int(version[0]) if version else None
    except ValueError:
        version = None

    try:
        return [_registry[(module, name, version)]]
    except KeyError:
        # Try dynamically importing (a) relative to the onnx_passes.passes
        # and (b) interpreting the module as a python module.
        try:
            m = import_module(f"onnx_passes.passes.{module}".strip("."))
            module = m.__name__
        except ModuleNotFoundError:
            try:
                import_module(f"{module}")
            except ModuleNotFoundError:
                pass

    # If no version is specified, try with the largest supported version
    # available in the registry
    if version is None:
        try:
            version = max(v for _, n, v in _registry.keys() if n == name)

            try:
                return [_registry[(module, name, version)]]
            except KeyError:
                pass
        except ValueError:
            pass

    # If name-version uniquely identifies a pass without the module, we can
    # resolve this
    passes = [
        cls for (_, n, v), cls in _registry.items() if
        n == name and v == version
    ]

    if len(passes) == 1:
        return passes

    # Does not seem to be a fully qualified pass identifier, but maybe this
    # refers to an entire module of passes to collect

    # Try dynamically importing (a) relative to the onnx_passes.passes
    # and (b) interpreting the identifier as a python module.
    try:
        m = import_module(f"onnx_passes.passes.{identifier}".strip("."))
        identifier = m.__name__
    except ModuleNotFoundError:
        try:
            import_module(f"{identifier}")
        except ModuleNotFoundError:
            if identifier not in {module for module, _, _ in _registry.keys()}:
                raise KeyError(f"Could not resolve passes: {identifier}")

    # Collect all passes registered within the module referred to by identifier
    passes = []

    for (module, name, version), cls in _registry.items():
        if module == identifier:
            passes.append(cls)

    if not passes:
        raise KeyError(f"Could not resolve passes: {identifier}")

    return passes


def resolve_module(identifier: str):
    """Resolves the module implementing a collection of passes."""

    # Try dynamically importing (a) relative to the onnx_passes.passes
    # and (b) interpreting the identifier as a python module.
    try:
        return import_module(f"onnx_passes.passes.{identifier}".strip("."))
    except ModuleNotFoundError:
        try:
            return import_module(f"{identifier}")
        except ModuleNotFoundError:
            return None


# Common cleanup passes already implemented in ONNX IR, used here without any
# custom infrastructure.
import onnx_ir.passes.common


class Transformation(Pass, ABC):
    """Base class for deriving transformation passes modifying the model."""

    @property
    def in_place(self) -> bool:
        return True

    @property
    def changes_input(self) -> bool:
        return True

    def cleanup(self, result: ir.passes.PassResult):
        """Performs cleanup of the pass result."""

        # Apply basic ONNX IR cleanup transformations to the model if the result
        # indicates the model to be modified, skip cleaning up unchanged models
        if result.modified:
            cleanup = ir.passes.PassManager([
                ir.passes.common.TopologicalSortPass(),
                ir.passes.common.RemoveUnusedNodesPass(),
                ir.passes.common.RemoveUnusedFunctionsPass(),
                ir.passes.common.RemoveUnusedOpsetsPass(),
                ir.passes.common.LiftConstantsToInitializersPass(),
                ir.passes.common.RemoveInitializersFromInputsPass(),
                ir.passes.common.DeduplicateInitializersPass(),
                ir.passes.common.ShapeInferencePass()
                # TODO: Give canonical names pass...
            ])

            return ir.passes.PassResult(cleanup(result.model).model, True)

        return result


class Sequential(Pass, ABC):
    """Sequence of passes, optionally applied exhaustive.

    Attributes
    ----------
    passes : list
        List of unresolved passes, can be classes, modules or string identifiers
    exhaustive : bool = False
        Apply th sequence exhaustively until the model stops changing
    """

    passes: list
    exhaustive: bool = False

    def __init__(self, config: Config = Config()):
        """Initializes the sequence by resolving and configuring the passes."""
        super().__init__(config)

        # Resolve all passes (each identifier from the list might resolve to a
        # collection of passes) and pass on the configuration
        self.passes = [
            cls(config) for p in self.passes for cls in resolve_passes(p)
        ]

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        """Applies the sequence of passes to the model."""

        # Wrap the sequence in a new pass manager to run the sequence and always
        # run the sequence at least once
        passes = ir.passes.PassManager(self.passes)
        result = passes(model)

        # If the composed pass is marked exhaustive, apply the sequence of
        # passes as long as there are changes to the model
        while self.exhaustive and result.modified:
            result = passes(result.model)

        return result


# Pattern-based graph rewriting implemented in ONNX Script
from onnxscript.rewriter import pattern, RewritePass, MatchResult


class RewriteRule(Transformation, ABC):
    """Base class for pattern-based rewrite rule transformation passes."""

    @abstractmethod
    def pattern(self, *args, **kwargs):
        """The target pattern to be matched."""
        raise NotImplementedError(
            "Method 'pattern' must be implemented by derived class."
        )

    @abstractmethod
    def rewrite(self, *args, **kwargs):
        """The replacement pattern to be inserted into the graph"""
        raise NotImplementedError(
            "Method 'rewrite' must be implemented by derived class."
        )

    def check(self, *args, **kwargs) -> MatchResult:  # noqa: static, unused arg
        """Match condition to decide whether to rewrite the matched pattern."""
        return MatchResult()

    @property
    def commute(self) -> bool:
        """Allow patterns of commutative ops to commute."""
        return False

    def rule(self):
        """Assembles the rewrite rule pass from the class definition."""

        def _check(op, *args, **kwargs):
            """Check to prevent rewriting inside functions."""
            if isinstance(op.graph_or_function, ir.Function):
                return False
            return self.check(op, *args, **kwargs)

        return pattern.RewriteRule(
            self.pattern, self.rewrite, _check, remove_nodes=False,
            verbose=self.config.logging.verbose
        )

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        """Applies the rewrite rule to the model."""
        rule_set = pattern.RewriteRuleSet([self.rule()], commute=self.commute)
        return RewritePass(rule_set)(model)


# Partial application of function used to partially modify pattern-based rule
# set definitions
from functools import partial


class RewriteRuleSet(Transformation, ABC):
    """Base class for pattern-based rewrite rule set transformation passes."""

    @abstractmethod
    def pattern(self):
        """List of target patterns to be matched."""
        raise NotImplementedError(
            "Method 'pattern' must be implemented by derived class."
        )

    @abstractmethod
    def rewrite(self):
        """The replacement patterns to be inserted into the graph"""
        raise NotImplementedError(
            "Method 'rewrite' must be implemented by derived class."
        )

    def check(self) -> list[Callable[..., MatchResult]]:
        """Match conditions to decide whether to rewrite a matched pattern."""
        return [
            lambda *args, **kwargs: MatchResult() for _ in self.pattern()
        ]

    @property
    def commute(self) -> bool:
        """Allow patterns of commutative ops to commute."""
        return False

    def rules(self):
        """Assembles the rewrite rule set pass from the class definition."""

        def _check(check, op, *args, **kwargs):  # noqa: Duplicate
            """Check to prevent rewriting inside functions."""
            if isinstance(op.graph_or_function, ir.Function):
                return False
            return check(op, *args, **kwargs)

        check = [partial(_check, check) for check in self.check()]
        rules = zip(self.pattern(), self.rewrite(), check)

        return [
            pattern.RewriteRule(
                *rule, remove_nodes=False,
                verbose=self.config.logging.verbose
            )
            for rule in rules
        ]

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        """Applies the rewrite rule set to the model."""
        rule_set = pattern.RewriteRuleSet(self.rules(), commute=self.commute)
        return RewritePass(rule_set)(model)
