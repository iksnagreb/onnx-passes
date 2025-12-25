# os.makedirs for creating logging directories on the fly
import os

# The base classes defined below are still not fully functional passes, but
# abstract bases themselves
import abc

# ir.Model, ir.save, ...
import onnx_ir as ir

# Base classes inherited from ONNX IR used by the custom ONNX passes
from onnx_ir.passes import PassBase, FunctionalPass, InPlacePass


# Base class for deriving all custom passes of the ONNX IR pass library: This
# adds configuration and state handling and serves as a marker type for building
# the registry of named/categorized passes.
class Pass(PassBase, abc.ABC):
    # Initializes a pass sets references to optional configuration and state
    # dictionary as instance attributes
    def __init__(self, config: dict | None, state: dict | None):
        self.config = config
        self.state_dict = state
        # Used by verification to inject expected outputs for post-condition
        self.expected = None
        self._id = None

    # Inject generating a unique pass-id available for all pre- and post-
    # conditions, as well as the wrapped __call__ and call methods
    def __call__(self, model_or_result: ir.Model | ir.passes.PassResult, /):
        # Count the number of passes already applied to the model to derive
        # unique checkpoint filenames
        i = self.state_dict.setdefault("counter", 0)
        # Generate a unique pass id valid until the next call to this pass
        self._id = f"{i:08d}-{type(self).__name__}"
        # Increment the step counter
        self.state_dict["counter"] += 1

        # ======================================================================
        # The following is essentially a copy of the PassBase.__call__ at the
        # time of onnx-ir==0.1.10, modified to forward the PassResult to the
        # post-condition, not just the model.
        # ======================================================================

        # PassBase accepts both, ir.Model and ir.passes.PassResult from previous
        # pass application in a sequence of passes
        if isinstance(model_or_result, ir.passes.PassResult):
            model = model_or_result.model
        else:
            model = model_or_result

        # Evaluate the precondition on the model, which raises an exception to
        # indicate failure
        try:
            self.requires(model)
        # Explicit precondition error is simply forwarded to the caller
        except ir.passes.PreconditionError:
            raise
        # All other exceptions are augmented to indicate that this is from
        # within the precondition
        except Exception as e:
            raise ir.passes.PreconditionError(
                f"Pre-condition for pass '{self.__class__.__name__}' failed"
            ) from e

        # Call the pass implementation (provided by the specialization) on the
        # model producing a PassResult
        result = self.call(model)

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

        # Evaluate the postcondition on the pass result (model and indication on
        # whether the model has been modified), which raises an exception to
        # indicate failure
        try:
            self.ensures(result)
        # Explicit postcondition error is simply forwarded to the caller
        except ir.passes.PostconditionError:
            raise
        # All other exceptions are augmented to indicate that this is from
        # within the postcondition
        except Exception as e:
            raise ir.passes.PostconditionError(
                f"Post-condition for pass '{self.__class__.__name__}' failed"
            ) from e

        # Forward the pass result to the derived implementations (or the caller
        # if this is the outermost specialization)
        return result

    # Unique pass-id to identify a pass across repeated applications within a
    # sequence of passes
    @property
    def id(self):
        return self._id

    # Pre-condition evaluated before entering a pass - implements verbosity
    def requires(self, model: ir.Model) -> None:
        # Verbosity can be enabled globally by setting it to True
        self.config.setdefault("logging", {}).setdefault("verbose", False)
        # Verbosity should now be defined, either defaulting to False or
        # explicitly
        if self.config["logging"]["verbose"]:
            # TODO: Make use of a proper logger...
            print(f"Entering {self.__class__.__name__}")

        # Model checkpointing can be disabled globally by setting the option to
        # False, otherwise it is interpreted as a filename to write the model
        # checkpoint to
        if self.config["logging"].setdefault("checkpoint", False):
            # Mark this as the before-the-pass checkpoint
            filename = f"before-{self.config['logging']['checkpoint']}"
            # Save the model checkpoint
            ir.save(model, filename)

    # Post-condition evaluated after leaving a pass - implements verbosity
    def ensures(self, result: ir.passes.PassResult) -> None:
        # Verbosity can be enabled globally by setting it to True
        self.config.setdefault("logging", {}).setdefault("verbose", False)
        # Verbosity should now be defined, either defaulting to False or
        # explicitly
        if self.config["logging"]["verbose"]:
            # TODO: Make use of a proper logger...
            print(f"Leaving {self.__class__.__name__}")

        # Model checkpointing can be disabled globally by setting the option to
        # False, otherwise it is interpreted as a filename to write the model
        # checkpoint to
        if self.config["logging"].setdefault("checkpoint", False):
            # Mark this as the after-the-pass checkpoint
            filename = f"after-{self.config['logging']['checkpoint']}"
            # Save the model checkpoint
            ir.save(result.model, filename)

        # Detailed logging of all intermediate models can be disabled globally
        # by setting the option to False, otherwise it is interpreted as a
        # pathname to write the models checkpoints to
        if self.config["logging"].setdefault("keep_intermediates", False):
            # Get the logging directory pathname
            path = self.config["logging"]["keep_intermediates"]
            # Make sure the directory exists...
            os.makedirs(path, exist_ok=True)
            # Mark this as the after-the-pass checkpoint
            filename = os.path.join(path, f"{self.id}.onnx")
            # Save the model checkpoint
            ir.save(result.model, filename)

        # Write a detailed history of passes finished on the model into the
        # state dictionary
        self.state_dict.setdefault("history", []).append(type(self))


# Base class for deriving analysis passes, which are side-effect-only passes,
# i.e., may only modify configuration and state dictionaries or other externally
# referenced objects (this includes printing/output), but not the model.
class Analysis(Pass, abc.ABC):
    @property
    def in_place(self) -> bool:
        return True

    @property
    def changes_input(self) -> bool:
        return False


# Base class for deriving annotation passes, which are in-place passes which
# modify the original model.
#
# Annotation passes *should* not modify the structure or any values contained in
# the model, only attributes, shapes or data types.
class Annotation(Pass, InPlacePass, abc.ABC):
    ...


# Node-removal pass build into ONNX IR and ONNX Script
from onnxscript.optimizer import remove_unused_nodes


# Base class for deriving transformation passes, which are in-place passes which
# modify the original model.
#
# Transformation passes may modify arbitrary properties of the model, including
# structure and values.
class Transformation(Pass, InPlacePass, abc.ABC):
    # There might be unused nodes after transforming parts of the graph, always
    # make sure to remove those before checking any other post-conditions - this
    # mostly prevents the output to be spammed with warning messages...
    def ensures(self, result: ir.passes.PassResult) -> None:
        super().ensures(result), remove_unused_nodes(result.model)


# Pattern-based graph rewriting implemented in ONNX Script
from onnxscript.rewriter import RewritePass
from onnxscript.rewriter.pattern import RewriteRule, RewriteRuleSet, MatchResult


# Base class for deriving pattern-based rewrite passes - when specialized must
# be mixed with either Annotation or Transformation as needed
class RewriteRulePass(Pass, abc.ABC):
    # Assemble a RewriteRule from the class definition: The specializing class
    # must implement the rules according to the base class RewriteRuleClassBase
    def rule(self):
        # Verbosity can be enabled globally by setting it to True
        v = self.config.setdefault("logging", {}).setdefault("verbose", False)
        # Extra arguments passed to the constructed rewrite rule: removing the
        # nodes for some reason prevents effective streamlining of residuals,
        # probably because the forking node has multiple consumer and thus
        # cannot be removed causing the pattern to be ignored even if matched.
        # TODO: Verify whether this is indeed the case and if so, whether this
        #  is intended behavior or a bug...
        kwargs = {"verbose": v, "remove_nodes": False}

        # Wrap the check to add extra global conditions which cannot be
        # overridden
        def _check(op, *_args, **_kwargs):
            # Do not apply rewrites inside functions as attribute references
            # will result in many checks trying to use Nones
            if isinstance(op.graph_or_function, ir.Function):
                return False
            # Use derived class check specialization
            return self.check(op, *_args, **_kwargs)

        # Inject bound(!) methods for detecting and replacing the pattern into
        # the rewrite rule
        return RewriteRule(self.pattern, self.rewrite, _check, **kwargs)

    @abc.abstractmethod
    def pattern(self, *args, **kwargs):
        raise NotImplementedError(
            "Method 'pattern' must be implemented by derived class.")

    @abc.abstractmethod
    def rewrite(self, *args, **kwargs):
        raise NotImplementedError(
            "Method 'rewrite' must be implemented by derived class.")

    def check(self, *args, **kwargs) -> MatchResult:
        return MatchResult()

    @property
    def commute(self) -> bool:
        return False

    # Implement the pass by assembling the pattern-based rewrite rule from the
    # class definition
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Create a rule set from the class definition and commutativity flag
        rule_set = RewriteRuleSet([self.rule()], commute=self.commute)
        # Apply the rule as the single rule of a rewrite pass
        return RewritePass(rule_set)(model)


# Partial application of function used to bind match conditions to the wrapper
# when setting up the rewrite rule set
from functools import partial


# Base class for deriving pattern-based rewrite passes from a set of rewrite
# rules - when specialized must be mixed with either Annotation or
# Transformation as needed
class RewriteRuleSetPass(Pass, abc.ABC):
    # Assemble list of RewriteRule from the class definition: The specializing
    # class must implement lists of matching rules returned by pattern and
    # rewrite (and optionally check) methods.
    def rules(self):
        # Verbosity can be enabled globally by setting it to True
        v = self.config.setdefault("logging", {}).setdefault("verbose", False)
        # Extra arguments passed to the constructed rewrite rule: removing the
        # nodes for some reason prevents effective streamlining of residuals,
        # probably because the forking node has multiple consumer and thus
        # cannot be removed causing the pattern to be ignored even if matched.
        # TODO: Verify whether this is indeed the case and if so, whether this
        #  is intended behavior or a bug...
        kwargs = {"verbose": v, "remove_nodes": False}

        # Wrap the check to add extra global conditions which cannot be
        # overridden
        def wrapper(check, op, *_args, **_kwargs):
            # Do not apply rewrites inside functions as attribute references
            # will result in many checks trying to use Nones
            if isinstance(op.graph_or_function, ir.Function):
                return False
            # Use derived class check specialization
            return check(op, *_args, **_kwargs)

        # Wrap each match condition: Partially bind the wrapper and collect all
        # checks as a list - cannot use a generator here
        _check = [partial(wrapper, check) for check in self.check()]

        # Create the list of rules by combining input and output pattern and the
        # condition
        rules = zip(self.pattern(), self.rewrite(), _check)
        # Inject methods for detecting and replacing the pattern into the
        # rewrite rule
        return [RewriteRule(*rule, **kwargs) for rule in rules]

    @abc.abstractmethod
    def pattern(self):
        raise NotImplementedError(
            "Method 'pattern' must be implemented by derived class.")

    @abc.abstractmethod
    def rewrite(self):
        raise NotImplementedError(
            "Method 'rewrite' must be implemented by derived class.")

    def check(self):
        return [
            lambda *args, **kwargs: MatchResult() for _ in self.pattern()
        ]

    @property
    def commute(self) -> bool:
        return False

    # Implement the pass by assembling the pattern-based rewrite rule from the
    # class definition
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Create a rule set from the class definition and commutativity flag
        rule_set = RewriteRuleSet(self.rules(), commute=self.commute)
        # Apply the rules as the rewrite pass
        return RewritePass(rule_set)(model)
