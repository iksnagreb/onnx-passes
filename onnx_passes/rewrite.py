from onnx_passes.base import Transformation

from abc import ABC
from typing import Any, Callable, Iterable

import onnx_ir as ir

# Pattern-based graph rewriting implemented in ONNX Script
import onnxscript.rewriter as rewriter


def getattr_v(obj: object, name: str, version: int) -> Any:
    """Get named attribute with ONNX-style version resolution."""
    for v in range(version, -1, -1):
        try:
            return getattr(obj, f"{name}_v{v}")
        except AttributeError:
            pass

    return getattr(obj, name)


class RewriteRule(Transformation, ABC):
    """Base class for pattern-based rewrite rule transformation passes."""

    def check(self, *args, **kwargs) -> rewriter.MatchResult:  # noqa: static
        """Match condition to decide whether to rewrite the matched pattern."""
        return rewriter.MatchResult()

    @property
    def commute(self) -> bool:
        """Allow patterns of commutative ops to commute."""
        return False

    def rule(self, model: ir.Model | None = None):
        """Assembles the rewrite rule pass from the class definition."""

        # Versioned lookup of rewrite rules if a model with an opset import for
        # the standard domain is given
        version = model.opset_imports[""] if model is not None else -1

        try:
            pattern = getattr_v(self, "pattern", version)
            check = getattr_v(self, "check", version)
            rewrite = getattr_v(self, "rewrite", version)
        except AttributeError:
            return None

        def _check(op, *args, **kwargs):
            """Check to prevent rewriting inside functions."""
            if isinstance(op.graph_or_function, ir.Function):
                return False
            return check(op, *args, **kwargs)

        rule = (pattern, rewrite, _check)

        return rewriter.RewriteRule(
            *rule, remove_nodes=False, verbose=self.config.logging.verbose
        )

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        """Applies the rewrite rule to the model."""
        if (rule := self.rule(model)) is not None:
            rule = rewriter.RewriteRuleSet([rule], commute=self.commute)
            return rewriter.RewritePass(rule)(model)

        if self.config.logging.verbose:
            print(
                f"Skipping {self.identifier}: No applicable rule for"
                f" v{model.opset_imports['']} of the standard opset"
            )

        return ir.passes.PassResult(model, False)


# Partial application of function used to partially modify pattern-based rule
# set definitions
from functools import partial


class RewriteRuleSet(Transformation, ABC):
    """Base class for pattern-based rewrite rule set transformation passes."""

    @staticmethod
    def check() -> Iterable[Callable[..., rewriter.MatchResult]]:
        """Match conditions to decide whether to rewrite a matched pattern."""
        return []

    @property
    def commute(self) -> bool:
        """Allow patterns of commutative ops to commute."""
        return False

    def rules(self, model: ir.Model | None = None):
        """Assembles the rewrite rule set pass from the class definition."""

        # Versioned lookup of rewrite rules if a model with an opset import for
        # the standard domain is given
        version = model.opset_imports[""] if model is not None else -1

        try:
            pattern = getattr_v(self, "pattern", version)
            check = getattr_v(self, "check", version)
            rewrite = getattr_v(self, "rewrite", version)
        except AttributeError:
            return None

        def _check(check, op, *args, **kwargs):  # noqa: Duplicate
            """Check to prevent rewriting inside functions."""
            if isinstance(op.graph_or_function, ir.Function):
                return False
            return check(op, *args, **kwargs)

        def default_check(op, *args, **kwargs):
            return True

        check = [partial(_check, check) for check in check()]

        pattern = list(pattern())
        rewrite = list(rewrite())

        if len(check) < len(pattern):
            check = [*check, *[partial(_check, default_check) for _ in pattern]]

        rules = zip(pattern, rewrite, check)

        return [
            rewriter.RewriteRule(
                *rule, remove_nodes=False, verbose=self.config.logging.verbose
            )
            for rule in rules
        ]

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        """Applies the rewrite rule set to the model."""
        if (rules := self.rules(model)) is not None:
            rules = rewriter.RewriteRuleSet(rules, commute=self.commute)
            return rewriter.RewritePass(rules)(model)

        if self.config.logging.verbose:
            print(
                f"Skipping {self.identifier}: No applicable rules for"
                f" v{model.opset_imports['']} of the standard opset"
            )

        return ir.passes.PassResult(model, False)


class RewriteRuleSetTemplate(RewriteRuleSet, ABC):
    """Template of pattern-based rewrite rule set transformation passes."""

    patterns: tuple
    rewrites: tuple | None = None

    def check(self, *args, **kwargs) -> rewriter.MatchResult:  # noqa: static
        """Match condition to decide whether to rewrite the matched pattern."""
        return rewriter.MatchResult()

    def rules(self, model: ir.Model | None = None):
        """Assembles the rewrite rule set pass from the class definition."""

        # Versioned lookup of rewrite rules if a model with an opset import for
        # the standard domain is given
        version = model.opset_imports[""] if model is not None else -1

        try:
            pattern = getattr_v(self, "pattern", version)
            check = getattr_v(self, "check", version)
            rewrite = getattr_v(self, "rewrite", version)
        except AttributeError:
            return None

        # Instantiate the pattern/check/rewrite templates for all patterns
        # listed in the template specialization
        patterns = self.__class__.patterns
        rewrites = self.__class__.rewrites

        if rewrites is None:
            rewrites = patterns

        if len(patterns) != len(rewrites):
            raise SyntaxError(
                f"Number of patterns and rewrites for instantiation of"
                f" RewriteRuleSetTemplate {self.identifier} does not match."
            )

        pattern = [partial(pattern, _pattern) for _pattern in patterns]
        rewrite = [partial(rewrite, _rewrite) for _rewrite in rewrites]

        check = len(patterns) * [check]

        def _check(check, op, *args, **kwargs):  # noqa: Duplicate
            """Check to prevent rewriting inside functions."""
            if isinstance(op.graph_or_function, ir.Function):
                return False
            return check(op, *args, **kwargs)

        check = [partial(_check, check) for check in check]

        rules = zip(pattern, rewrite, check)

        return [
            rewriter.RewriteRule(
                *rule, remove_nodes=False, verbose=self.config.logging.verbose
            )
            for rule in rules
        ]


class RewriteAsConstant(RewriteRule, ABC):
    """Template to rewrite a pattern by expanding a matching constant.

    Instantiations of the template must implement the pattern according to the
    RewriteRule with an _outputs=["out"] matched on the final node.
    """

    constant: Any

    @staticmethod
    def check(context, *args, out, **kwargs):
        if out.shape is not None and out.shape.is_static():
            return out.dtype is not None
        return False

    def rewrite(self, op, *args, out, **kwargs):
        return op.Expand(
            op.Cast(
                op.Constant(value=ir.tensor(self.constant)),
                to=out.dtype.value
            ),
            op.Constant(
                value_ints=ir.Attr(
                    "value_ints", ir.AttributeType.INTS, out.shape[:]
                )
            )
        )
