"""F5 Rule 6: require one connected input-to-digest route."""

from __future__ import annotations

import ast
import re
from typing import TYPE_CHECKING

from crypto_steering.validation.java.task_analysis import (
    _factory_receivers_assigned_from,
    _receiver_method_calls,
)
from crypto_steering.validation.models import TaskEvidence, TaskValidationResult
from crypto_steering.validation.source.index import SourceAnalyzer, analyzer_for
from crypto_steering.validation.tasks.shared.results import task_failure
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.evidence import (
    _digest_comparison_evidence,
)
from crypto_steering.validation.tasks.shared.stateful import repeated_terminal_receiver

if TYPE_CHECKING:
    from crypto_steering.validation.tasks.f05.api_family import Analysis


def validate_rule(
    context: TaskContext,
    analysis: "Analysis",
    result: TaskValidationResult,
) -> TaskValidationResult | None:
    """Comparison and downstream result use are outside family validity."""

    return None


def validate_java_rule(
    context: TaskContext,
    analysis: "Analysis",
    result: TaskValidationResult,
) -> TaskValidationResult | None:
    if (
        context.language == "java"
        and not analysis.chained_digest
        and not analysis.completed_digest
        and analysis.has_digest_terminal
    ):
        return task_failure(
            result,
            "target_operation_sequence_invalid",
            "F5 requires MessageDigest.digest on input data or after update",
        )
    return None


def validate_python_rule(
    context: TaskContext,
    analysis: "Analysis",
    result: TaskValidationResult,
) -> TaskValidationResult | None:
    """Apply Python-only connected digest-route checks after early arguments."""

    if context.language != "python":
        return None
    failure = _python_empty_digest_failure(context.source)
    if failure is None:
        failure = _python_unusable_digest_route(context.source)
    if failure is None:
        return None
    return task_failure(result, "target_operation_sequence_invalid", failure)


def _python_empty_digest_failure(source: str) -> str | None:
    """Reject finalizing a newly constructed hash before supplying any data."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if not (
            isinstance(call.func, ast.Attribute)
            and call.func.attr in {"digest", "hexdigest"}
            and isinstance(call.func.value, ast.Call)
        ):
            continue
        constructor = call.func.value
        name = _python_ast_name(constructor.func)
        if (
            isinstance(constructor.func, ast.Call)
            and _python_ast_name(constructor.func.func) == "getattr"
            and constructor.func.args
            and _python_ast_name(constructor.func.args[0]) == "hashlib"
            and not constructor.args
            and not constructor.keywords
        ):
            return "the digest is finalized before the task data is supplied"
        if name == "hashlib.new" and len(constructor.args) < 2 and not any(
            keyword.arg in {"data", "string"} for keyword in constructor.keywords
        ):
            return "the digest is finalized before the task data is supplied"
        if (
            name.rsplit(".", 1)[-1].lower()
            in {
                "md5",
                "sha1",
                "sha224",
                "sha256",
                "sha384",
                "sha512",
                "sha3_224",
                "sha3_256",
                "sha3_384",
                "sha3_512",
                "blake2b",
                "blake2s",
            }
            and not constructor.args
            and not constructor.keywords
        ):
            return "the digest is finalized before the task data is supplied"
    return None


def _python_unusable_digest_route(source: str) -> str | None:
    """Reject definite receiver and callable errors on the selected hash route."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    if any(
        isinstance(call, ast.Call)
        and _python_ast_name(call.func) == "os.strencode"
        for call in ast.walk(tree)
    ):
        return "the os module has no strencode operation"

    class_text_failure = _python_class_digest_text_input_failure(tree)
    if class_text_failure is not None:
        return class_text_failure

    hash_factories: set[str] = set()
    text_values: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            names = tuple(
                target.id for target in targets if isinstance(target, ast.Name)
            )
            if (
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id == "hashlib"
                and re.fullmatch(
                    r"(?:md5|sha(?:1|224|256|384|512)|sha3_(?:224|256|384|512)|blake2[bs])",
                    value.attr,
                )
            ):
                hash_factories.update(names)
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                text_values.update(names)

    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        call_name = _python_ast_name(call.func)
        if re.fullmatch(
            r"hashlib\.(?:md5|sha(?:1|224|256|384|512)|sha3_(?:224|256|384|512)|blake2[bs])",
            call_name,
        ) and call.args:
            value = call.args[0]
            if (
                isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            ) or (isinstance(value, ast.Name) and value.id in text_values):
                return "hashlib digest constructors require bytes-like input, not text"
        if (
            isinstance(call.func, ast.Attribute)
            and call.func.attr in {"digest", "hexdigest"}
            and call.args
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id in hash_factories
        ):
            return (
                "a hashlib constructor is used as a hash object; construct it "
                "before calling digest"
            )
    return None


def _python_class_digest_text_input_failure(tree: ast.Module) -> str | None:
    """Follow constructor text into a ``self`` attribute hashed as bytes."""

    module_text = {
        target.id
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
        if isinstance(target, ast.Name)
    }

    def definitely_text(node: ast.AST) -> bool:
        return bool(
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            or isinstance(node, ast.Name)
            and node.id in module_text
        )

    for class_node in (node for node in tree.body if isinstance(node, ast.ClassDef)):
        initializer = next(
            (
                node
                for node in class_node.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "__init__"
            ),
            None,
        )
        if initializer is None:
            continue
        parameters = tuple(initializer.args.posonlyargs) + tuple(initializer.args.args)
        parameter_indexes = {
            parameter.arg: index - 1
            for index, parameter in enumerate(parameters)
            if index > 0
        }
        attribute_parameters = {
            target.attr: node.value.id
            for node in ast.walk(initializer)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and isinstance(node.value, ast.Name)
            for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
            if isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and node.value.id in parameter_indexes
        }
        if not attribute_parameters:
            continue
        text_attributes: set[str] = set()
        for call in (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and _python_ast_name(node.func).rsplit(".", 1)[-1] == class_node.name
        ):
            for attribute, parameter in attribute_parameters.items():
                position = parameter_indexes[parameter]
                argument = call.args[position] if position < len(call.args) else next(
                    (
                        keyword.value
                        for keyword in call.keywords
                        if keyword.arg == parameter
                    ),
                    None,
                )
                if argument is not None and definitely_text(argument):
                    text_attributes.add(attribute)
        for call in (node for node in ast.walk(class_node) if isinstance(node, ast.Call)):
            if not re.fullmatch(
                r"hashlib\.(?:md5|sha(?:1|224|256|384|512)|sha3_(?:224|256|384|512)|blake2[bs])",
                _python_ast_name(call.func),
            ) or not call.args:
                continue
            argument = call.args[0]
            if (
                isinstance(argument, ast.Attribute)
                and isinstance(argument.value, ast.Name)
                and argument.value.id == "self"
                and argument.attr in text_attributes
            ):
                return "hashlib digest constructors require bytes-like input, not text"
    return None


def _python_ast_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _python_ast_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""




def _typescript_first_digest_result_is_compared(
    source: str,
    receiver: str,
) -> bool:
    """Allow later diagnostic reuse after a completed integrity comparison."""

    analyzer = analyzer_for("typescript", source)
    digest_results = tuple(
        assignment
        for assignment in getattr(analyzer, "assignments", ())
        if re.search(
            rf"(?<![\w$]){re.escape(receiver)}\s*\.\s*digest\s*\(",
            assignment.expression,
        )
    )
    if not digest_results:
        return False
    first = digest_results[0]
    if not any(
        call.start < first.start
        and re.sub(r"\s+", "", call.callee).startswith(receiver + ".")
        and call.arguments
        for call in analyzer.calls("update")
    ):
        return False
    result = re.sub(r"\s+", "", first.name)
    comparisons = (
        rf"\b{re.escape(result)}\b\s*(?:===|!==|==|!=)\s*[^;\n]+",
        rf"[^;\n]+\s*(?:===|!==|==|!=)\s*\b{re.escape(result)}\b",
        rf"(?:timingSafeEqual|equals|compare)\s*\([^)]*\b{re.escape(result)}\b",
    )
    return any(re.search(pattern, source, flags=re.IGNORECASE) for pattern in comparisons)


def _comparison_evidence_for_language(
    language: str, source: str
) -> TaskEvidence | None:
    """Require a comparison that names a digest and an expected value."""

    return _digest_comparison_evidence(  # type: ignore[arg-type]
        analyzer_for(language, source)
    )


def _digest_comparison_failure(language: str, source: str) -> str | None:
    """Reject comparisons that cannot decide the computed digest result."""

    if language == "python" and _python_digest_is_compared_with_itself(source):
        return "the computed digest is compared with itself, not the expected digest"

    if language == "java":
        java_failure = _java_digest_state_failure(source)
        if java_failure is not None:
            return java_failure

    if language == "typescript" and _typescript_compares_recomputed_digest(source):
        return (
            "the supplied expected digest is ignored and the computed digest "
            "is compared with a recomputation of the same input"
        )

    if language == "csharp" and re.search(
        r"[\"'](?:Is\s+)?hash[^\n;]*[\"']\s*\+\s*"
        r"[A-Za-z_$][\w$]*\s*==",
        source,
        flags=re.IGNORECASE,
    ):
        return (
            "string concatenation is evaluated before equality, so the "
            "computed digest is not compared with the expected digest"
        )
    discarded_digest_assignments = re.finditer(
        r"\b(?:var|byte\s*\[\s*\])\s+(?P<name>[A-Za-z_]\w*)\s*=\s*"
        r"[A-Za-z_]\w*\s*\.\s*TransformFinalBlock\s*\(",
        source,
        flags=re.IGNORECASE,
    )
    if language == "csharp" and any(
        re.search(r"(?:hash|digest|checksum)", match.group("name"), re.IGNORECASE)
        for match in discarded_digest_assignments
    ):
        return (
            "TransformFinalBlock returns the final input block, not the "
            "computed digest"
        )
    byte_arrays = {
        match.group("name")
        for match in re.finditer(
            r"\bbyte\s*\[\s*\]\s+(?P<name>[A-Za-z_]\w*)\b",
            source,
        )
    }
    for match in re.finditer(
        r"\b(?P<left>[A-Za-z_]\w*)\s*(?:==|!=)\s*"
        r"(?P<right>[A-Za-z_]\w*)\b",
        source,
    ):
        left, right = match.group("left"), match.group("right")
        if {left, right} <= byte_arrays and re.search(
            r"(?:hash|digest|checksum)", left + right, flags=re.IGNORECASE
        ):
            return "C# byte-array equality does not compare digest contents"
    if re.search(
        r"\b[A-Za-z_]\w*\s*\.\s*Hash\s*(?:==|!=)\s*"
        r"[A-Za-z_]\w*",
        source,
        flags=re.IGNORECASE,
    ):
        return "C# byte-array equality does not compare digest contents"
    if language == "csharp" and _csharp_compares_raw_content_to_digest(source):
        return (
            "file-integrity comparison uses raw content instead of a digest "
            "computed from that content"
        )
    return None


def _java_digest_state_failure(source: str) -> str | None:
    """Reject comparisons that finalize an unpopulated/reset digest state."""

    if re.search(
        r"MessageDigest\s*\.\s*getInstance\s*\([^;]+\)\s*\.\s*update\s*\(",
        source,
        flags=re.IGNORECASE,
    ) and re.search(
        r"(?:expected|actual|calculated|computed|hash|digest)[A-Za-z0-9_$]*\s*=\s*"
        r"MessageDigest\s*\.\s*getInstance\s*\([^;]+\)\s*\.\s*digest\s*\(\s*\)",
        source,
        flags=re.IGNORECASE,
    ):
        return "digest updates and finalization use different MessageDigest instances"

    analyzer = analyzer_for("java", source)
    if _java_hashes_location_text_instead_of_content(analyzer):
        return "the digest is computed from a resource location instead of its content"
    by_receiver: dict[str, list[object]] = {}
    for call in analyzer.calls("digest"):
        compact = re.sub(r"\s+", "", call.callee)
        if "." not in compact or call.arguments:
            continue
        by_receiver.setdefault(compact.rsplit(".", 1)[0], []).append(call)
    for receiver, calls in by_receiver.items():
        calls.sort(key=lambda item: item.start)
        for first, second in zip(calls, calls[1:]):
            updated_between = any(
                first.end < update.start < second.start
                and re.sub(r"\s+", "", update.callee).startswith(receiver + ".")
                for update in analyzer.calls("update")
            )
            if not updated_between:
                return (
                    f"MessageDigest {receiver} is finalized twice without new input; "
                    "the second digest is not the expected file digest"
                )
    return None


def _java_hashes_location_text_instead_of_content(analyzer: SourceAnalyzer) -> bool:
    """Find a file-integrity route that hashes only a URL/path string."""

    digest_receivers = _factory_receivers_assigned_from(
        analyzer.source,
        "MessageDigest",
    )
    if not digest_receivers:
        return False
    inputs: list[str] = []
    for operation in ("update", "digest"):
        for receiver, call in _receiver_method_calls(analyzer.source, operation):
            if receiver in digest_receivers and call.arguments:
                inputs.extend(argument.text for argument in call.arguments)
    if not inputs:
        return False
    location_names = {
        match.group("name")
        for match in re.finditer(
            r"\b(?:URL|URI|Path|File)\s+(?P<name>[A-Za-z_$][\w$]*)\s*=",
            analyzer.source,
        )
    }
    if not location_names:
        return False
    location = re.compile(
        rf"\b(?:{'|'.join(map(re.escape, sorted(location_names)))})\b"
        r"\s*(?:\.\s*toString\s*\(\s*\))?\s*\.\s*getBytes\s*\(",
        re.IGNORECASE,
    )
    return all(location.search(value) is not None for value in inputs)


def _typescript_compares_recomputed_digest(source: str) -> bool:
    """Find a digest compared only with another digest of the same input."""

    analyzer = analyzer_for("typescript", source)
    origins: dict[str, str] = {}
    for assignment in getattr(analyzer, "assignments", ()):
        expression = assignment.expression
        if re.search(r"\bcreateHash\s*\(", expression) is None or re.search(
            r"\.\s*digest\s*\(", expression
        ) is None:
            continue
        update = re.search(r"\.\s*update\s*\(\s*(?P<input>[^,)]+)", expression)
        if update is not None:
            origins[assignment.name] = re.sub(r"\s+", "", update.group("input"))
    if len(origins) < 2:
        return False
    return any(
        match.group("left") in origins
        and match.group("right") in origins
        and origins[match.group("left")] == origins[match.group("right")]
        for match in re.finditer(
            r"\b(?P<left>[A-Za-z_$][\w$]*)\s*(?:===|!==|==|!=)\s*"
            r"(?P<right>[A-Za-z_$][\w$]*)\b",
            source,
        )
    )


def _python_digest_is_compared_with_itself(source: str) -> bool:
    """Find a digest-like value used as both operands of one comparison."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], (ast.Eq, ast.NotEq))
        and len(node.comparators) == 1
        and isinstance(node.left, ast.Name)
        and isinstance(node.comparators[0], ast.Name)
        and node.left.id == node.comparators[0].id
        and re.search(
            r"(?:hash|digest|checksum|expected)",
            node.left.id,
            re.IGNORECASE,
        )
        for node in ast.walk(tree)
    )


def _csharp_compares_raw_content_to_digest(source: str) -> bool:
    """Recognize the definite raw-content-versus-digest wiring error.

    This check is intentionally data-flow scoped: it only rejects a comparison
    where one named operand comes directly from a byte-content reader, the
    other comes from a digest operation, and no digest operation consumes the
    content operand.
    """

    analyzer = analyzer_for("csharp", source)
    assignments = tuple(getattr(analyzer, "assignments", ()))
    digest_helpers = {
        match.group("name")
        for match in re.finditer(
            r"\b(?:byte\s*\[\s*\]|string|var)\s+"
            r"(?P<name>(?:Compute|Calculate|Generate|Create)[A-Za-z0-9_]*"
            r"(?:Hash|Digest|Checksum)[A-Za-z0-9_]*)\s*\(",
            source,
            flags=re.IGNORECASE,
        )
    }
    digest_expression = re.compile(
        r"\b(?:HashData|ComputeHash|GetHashAndReset)\s*\(|"
        + "|".join(
            rf"\b{re.escape(name)}\s*\(" for name in sorted(digest_helpers)
        )
        if digest_helpers
        else r"\b(?:HashData|ComputeHash|GetHashAndReset)\s*\("
    )
    raw_content = {
        _csharp_simple_name(assignment.name)
        for assignment in assignments
        if re.search(
            r"\b(?:ReadAllBytes|ReadAsByteArrayAsync|DownloadData)\s*\(",
            assignment.expression,
        )
        and not digest_expression.search(assignment.expression)
        and not re.search(
            r"(?:hash|digest|checksum)", assignment.name, flags=re.IGNORECASE
        )
    }
    raw_content.discard("")
    if not raw_content:
        return False

    digests = {
        _csharp_simple_name(assignment.name)
        for assignment in assignments
        if digest_expression.search(assignment.expression)
    }
    digests.discard("")
    if not digests:
        return False

    hashed_content = {
        raw_name
        for raw_name in raw_content
        if any(
            digest_expression.search(assignment.expression)
            and re.search(rf"\b{re.escape(raw_name)}\b", assignment.expression)
            for assignment in assignments
        )
    }
    if raw_content <= hashed_content:
        return False

    comparison_names = {
        "areequal",
        "arehashesequal",
        "comparebytearrays",
        "equals",
        "fixedtimeequals",
        "sequenceequal",
        "structuralequals",
    }
    for call in analyzer.calls():
        if call.simple_name.lower() not in comparison_names:
            continue
        operands = {
            _csharp_simple_name(argument) for argument in call.arguments
        }
        compact_callee = re.sub(r"\s+", "", call.callee)
        if "." in compact_callee:
            operands.add(_csharp_simple_name(compact_callee.rsplit(".", 1)[0]))
        if operands & (raw_content - hashed_content) and operands & digests:
            return True
    return False


def _csharp_simple_name(expression: str) -> str:
    compact = re.sub(r"\s+", "", expression)
    match = re.fullmatch(r"(?:this\.)?([A-Za-z_]\w*)", compact)
    return match.group(1) if match else ""
