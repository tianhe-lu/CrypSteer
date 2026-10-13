"""Optional diagnostics for deterministic runtime-invalid source shapes.

These checks are deliberately separate from primary structural validity.
"""

from __future__ import annotations

import ast
import re

from crypto_steering.validation.models import Language
from crypto_steering.validation.source.index import CallExpression, analyzer_for
from crypto_steering.validation.tasks.contracts import (
    _python_call_name,
    _python_cryptography_argument_failure,
    _typescript_callback_question_missing_query,
)


def _known_native_runtime_failure(
    language: Language,
    source: str,
) -> str | None:
    """Return a separate diagnostic for deterministic runtime-invalid shapes.

    This helper is intentionally not called by ``validate_target_task`` or by
    the primary validity predicate.
    """

    if language == "typescript":
        malformed_question = _typescript_callback_question_missing_query(source)
        if malformed_question is not None:
            return malformed_question
        if (
            re.search(r"\bscryptSync\s*\(", source)
            and re.search(r"\bN\s*:\s*(?:32768|2\s*\*\*\s*15)\b", source)
            and not re.search(r"\bmaxmem\s*:", source)
        ):
            return (
                "Node scrypt with N=32768 requires an explicit sufficient "
                "maxmem setting in this dataset toolchain"
            )
        if re.search(r"\bpadding\s*:\s*4\b", source):
            return "numeric RSA signature padding 4 is not a supported Node signing option"
        finalized_receiver = _typescript_repeated_final_receiver(source)
        if finalized_receiver is not None:
            return (
                f"{finalized_receiver}.final() is called more than once "
                "without constructing a replacement cipher"
            )
    if language == "python":
        if re.search(r"\brandom\s*\.\s*randbelow\s*\(", source):
            return (
                "random.randbelow is not a Python random-module API; "
                "use secrets.randbelow or random.randrange"
            )
        cryptography_failure = _python_cryptography_runtime_failure(source)
        if cryptography_failure is not None:
            return cryptography_failure
        local_call_failure = _python_local_call_arity_failure(source)
        if local_call_failure is not None:
            return local_call_failure
        if re.search(
            r"\bbytes\s*\.\s*fromhex\s*\(\s*"
            r"(?:base64\s*\.\s*)?b64decode\s*\(",
            source,
        ):
            return "bytes.fromhex cannot consume bytes returned by base64.b64decode"
        if re.search(
            r"\bfrom\s+cryptography\s*\.\s*exceptions\s+import\s+[^\n]*\bInvalidToken\b",
            source,
        ):
            return "InvalidToken is provided by cryptography.fernet, not cryptography.exceptions"
        if re.search(
            r"\bAESGCM\s*\.\s*generate_key\s*\(\s*bit_length\s*=\s*96\s*\)",
            source,
        ):
            return "AESGCM.generate_key does not accept a 96-bit key length"
        if re.search(
            r"\bHMAC\s*\(\s*[^,]+,\s*hashlib\s*\.\s*sha(?:256|384|512)\s*\(\s*\)\s*\)",
            source,
            flags=re.DOTALL,
        ):
            return (
                "the HMAC call supplies a hashlib object where the selected "
                "API requires either a PyCA hash algorithm or a digestmod argument"
            )
        invalid_constructor = _python_argumentful_class_without_constructor(source)
        if invalid_constructor is not None:
            return (
                f"local class {invalid_constructor} has no constructor but is "
                "instantiated with arguments"
            )
    return None


def _python_local_call_arity_failure(source: str) -> str | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    definitions: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(parents.get(node), ast.ClassDef):
            continue
        definitions.setdefault(node.name, []).append(node)

    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if not isinstance(call.func, ast.Name) or call.func.id not in definitions:
            continue
        if any(isinstance(argument, ast.Starred) for argument in call.args):
            continue
        if any(keyword.arg is None for keyword in call.keywords):
            continue
        keyword_names = {keyword.arg for keyword in call.keywords}
        if any(
            _python_call_matches_function_arity(
                call,
                function,
                keyword_names,
            )
            for function in definitions[call.func.id]
        ):
            continue
        return f"local function {call.func.id} is called with incompatible arguments"
    return None


def _python_call_matches_function_arity(
    call: ast.Call,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    keyword_names: set[str | None],
) -> bool:
    positional = tuple(function.args.posonlyargs) + tuple(function.args.args)
    required_count = len(positional) - len(function.args.defaults)
    supplied_positionally = len(call.args)
    if supplied_positionally > len(positional) and function.args.vararg is None:
        return False
    for position, parameter in enumerate(positional[:required_count]):
        if position < supplied_positionally:
            continue
        if parameter.arg not in keyword_names:
            return False
    if any(
        default is None and parameter.arg not in keyword_names
        for parameter, default in zip(
            function.args.kwonlyargs,
            function.args.kw_defaults,
            strict=True,
        )
    ):
        return False
    return True


def _python_cryptography_runtime_failure(source: str) -> str | None:
    """Recognize definite PyCA call-signature failures in generated code."""

    return _python_cryptography_argument_failure(source, families=None)


def _typescript_repeated_final_receiver(source: str) -> str | None:
    """Find a sequential second finalization of the same Node cipher."""

    try:
        analyzer = analyzer_for("typescript", source)
    except Exception:
        return None
    assignments = tuple(getattr(analyzer, "assignments", ()))
    by_receiver: dict[tuple[str | None, str], list[CallExpression]] = {}
    for call in analyzer.calls("final"):
        compact = re.sub(r"\s+", "", call.callee)
        if "." not in compact:
            continue
        receiver = compact.rsplit(".", 1)[0]
        key = (analyzer.containing_function_name(call.start), receiver)
        by_receiver.setdefault(key, []).append(call)
    for (_function, receiver), calls in by_receiver.items():
        ordered = sorted(calls, key=lambda item: item.start)
        for first, second in zip(ordered, ordered[1:]):
            reassigned = any(
                first.end < assignment.start < second.start
                and re.sub(r"\s+", "", assignment.name) == receiver
                for assignment in assignments
            )
            if reassigned:
                continue
            between = source[first.end : second.start]
            if re.search(r"\b(?:else|case|catch)\b", between):
                continue
            return receiver
    return None


def _python_argumentful_class_without_constructor(source: str) -> str | None:
    """Find a definite local ``Class(args)`` runtime arity error."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    classes = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
    }
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if not isinstance(call.func, ast.Name) or not (call.args or call.keywords):
            continue
        definition = classes.get(call.func.id)
        if definition is None or definition.bases:
            continue
        if any(
            _python_call_name(
                decorator.func if isinstance(decorator, ast.Call) else decorator
            ).rsplit(".", 1)[-1]
            == "dataclass"
            for decorator in definition.decorator_list
        ):
            continue
        if any(
            isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name in {"__init__", "__new__"}
            for item in definition.body
        ):
            continue
        return definition.name
    return None
