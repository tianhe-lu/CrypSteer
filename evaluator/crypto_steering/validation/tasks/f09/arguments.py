"""F9 Rule 5: validate the selected random-generation call."""

from __future__ import annotations

import ast

from crypto_steering.validation.models import TaskEvidence, TaskValidationResult
from crypto_steering.validation.source.python import PythonAnalyzer
from crypto_steering.validation.tasks.evidence import _PYTHON_RANDOM_CALLABLES, _python_random_call_target
from crypto_steering.validation.tasks.shared.python_callables import callable_aliases
from crypto_steering.validation.tasks.shared.results import require_analyzer, task_failure
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.f09.api_family import Analysis


_PYTHON_CALL_SHAPES: dict[str, tuple[int, int, frozenset[str]]] = {
    "random": (0, 0, frozenset()),
    "uuid4": (0, 0, frozenset()),
    "urandom": (1, 1, frozenset({"size"})),
    "token_bytes": (0, 1, frozenset({"nbytes"})),
    "token_hex": (0, 1, frozenset({"nbytes"})),
    "token_urlsafe": (0, 1, frozenset({"nbytes"})),
    "getrandbits": (1, 1, frozenset({"k"})),
    "randbits": (1, 1, frozenset({"k"})),
    "randbelow": (1, 1, frozenset({"exclusive_upper_bound"})),
    "randbytes": (1, 1, frozenset({"n"})),
    "choice": (1, 1, frozenset({"seq"})),
    "choices": (
        1,
        3,
        frozenset({"population", "weights", "cum_weights", "k"}),
    ),
    "randint": (2, 2, frozenset({"a", "b"})),
    "randrange": (1, 3, frozenset({"start", "stop", "step"})),
    "uniform": (2, 2, frozenset({"a", "b"})),
}

_PYTHON_POSITIONAL_ROLES: dict[str, tuple[str, ...]] = {
    "urandom": ("size",),
    "token_bytes": ("nbytes",),
    "token_hex": ("nbytes",),
    "token_urlsafe": ("nbytes",),
    "getrandbits": ("k",),
    "randbits": ("k",),
    "randbelow": ("exclusive_upper_bound",),
    "randbytes": ("n",),
    "choice": ("seq",),
    "choices": ("population", "weights"),
    "randint": ("a", "b"),
    "randrange": ("start", "stop", "step"),
    "uniform": ("a", "b"),
}


def validate_rule(
    context: TaskContext,
    analysis: Analysis,
    result: TaskValidationResult,
) -> TaskValidationResult | None:
    if context.language != "python" or analysis.generation is None:
        # The compiled frontends establish native call signatures for Java,
        # TypeScript, and C#. Python needs the narrow source-level check below.
        return None
    analyzer = require_analyzer(context)
    import_aliases = getattr(analyzer, "import_aliases", {})
    # Evidence discovery already recognizes native callables stored in object
    # state. Check their native signatures too: ``self.random = os.urandom``
    # does not give ``self.random(n)`` the zero-argument random.random contract.
    aliases = callable_aliases(
        context.source, import_aliases, _PYTHON_RANDOM_CALLABLES
    )

    def generation_failure(generation: TaskEvidence) -> str | None:
        names = dict(aliases)
        if isinstance(analyzer, PythonAnalyzer) and generation.span is not None:
            try:
                call = ast.parse(generation.expression.strip(), mode="eval").body
            except SyntaxError:
                call = None
            if isinstance(call, ast.Call):
                target = ast.unparse(call.func)
                canonical = _python_random_call_target(analyzer, target, generation.span.start)
                if canonical is not None:
                    names[_call_name(call.func)] = canonical
        return _python_random_argument_failure(
            generation.expression, import_aliases=import_aliases,
            callable_names=names,
        )

    failures = tuple(
        generation_failure(generation)
        for generation in analysis.generations
    )
    if not failures or any(reason is None for reason in failures):
        return None
    return task_failure(
        result,
        "target_argument_contract_invalid",
        next(reason for reason in failures if reason is not None),
    )


def _python_random_argument_failure(
    expression: str,
    *,
    import_aliases: dict[str, str] | None = None,
    callable_names: dict[str, str] | None = None,
) -> str | None:
    try:
        node = ast.parse(expression.strip(), mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(node, ast.Call):
        return None
    callee = _call_name(node.func)
    head, separator, tail = callee.partition(".")
    canonical = (import_aliases or {}).get(head, head) + (
        separator + tail if separator else ""
    )
    canonical = (callable_names or {}).get(callee, canonical)
    name = canonical.rsplit(".", 1)[-1]
    shape = _PYTHON_CALL_SHAPES.get(name)
    if shape is None:
        return None

    minimum, maximum, allowed_keywords = shape
    keyword_names = tuple(
        keyword.arg for keyword in node.keywords if keyword.arg is not None
    )
    unsupported = set(keyword_names) - allowed_keywords
    if unsupported:
        return f"{name} received unsupported keyword {sorted(unsupported)[0]}"
    if any(keyword.arg is None for keyword in node.keywords):
        # A dynamically expanded mapping cannot be resolved statically. It may
        # provide a usable call, so leave that runtime uncertainty outside the
        # structural validator.
        return None

    duplicated = set(
        _PYTHON_POSITIONAL_ROLES.get(name, ())[: len(node.args)]
    ) & set(keyword_names)
    if duplicated:
        return f"{name} supplies {sorted(duplicated)[0]} more than once"

    supplied = len(node.args) + len(keyword_names)
    if supplied < minimum:
        return f"{name} is missing a required random-generation argument"
    if supplied > maximum:
        return f"{name} received too many random-generation arguments"
    return None


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""
