"""F5 Rule 3: recognize standalone cryptographic digest APIs."""

from __future__ import annotations

from dataclasses import dataclass
import re

from crypto_steering.validation.java.source import find_calls, strip_java_comments
from crypto_steering.validation.java.task_analysis import (
    _digest_receivers_used_by_digest_input_stream,
    _factory_receivers_assigned_from,
    _receiver_method_calls,
)
from crypto_steering.validation.models import TaskEvidence, TaskValidationResult
from crypto_steering.validation.source.index import SourceAnalyzer
from crypto_steering.validation.tasks.shared.results import java_task_result, require_analyzer
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.evidence import (
    _call_evidence,
    _first_evidence,
    _hash_constructor_input_evidence,
    _is_python_hash_callable,
    _is_python_hashlib_callable,
    _python_hash_callable_evidence,
    _python_hash_terminal_evidence,
    _qualified_call_evidence,
)


@dataclass(frozen=True)
class Analysis:
    source: str
    required: tuple[tuple[TaskEvidence | None, str], ...] = ()
    chained_digest: bool = False
    digest_receivers: frozenset[str] = frozenset()
    completed_digest: bool = False
    has_digest_terminal: bool = False


def analyze(context: TaskContext) -> Analysis:
    if context.language == "java":
        source = strip_java_comments(context.source)
        chained = any(
            call.arguments
            and re.search(
                r"\bMessageDigest\s*\.\s*getInstance\s*\([^;{}]*\)\s*$",
                source[: call.start],
                flags=re.DOTALL,
            )
            for call in find_calls(
                source,
                r"\.\s*digest\s*\(",
                "MessageDigest.getInstance.digest",
            )
        )
        receivers = frozenset(
            _factory_receivers_assigned_from(source, "MessageDigest")
        )
        updated = {
            receiver
            for receiver, call in _receiver_method_calls(source, "update")
            if receiver in receivers and call.arguments
        }
        updated.update(
            _digest_receivers_used_by_digest_input_stream(source, set(receivers))
        )
        terminals = tuple(_receiver_method_calls(source, "digest"))
        completed = any(
            receiver in receivers and (call.arguments or receiver in updated)
            for receiver, call in terminals
        )
        return Analysis(
            source=source,
            chained_digest=chained,
            digest_receivers=receivers,
            completed_digest=completed,
            has_digest_terminal=bool(terminals),
        )

    analyzer = require_analyzer(context)
    setup = _first_evidence(
        analyzer,
        (
            "createHash", "HashData", "ComputeHash", "Hash", "md5", "sha1",
            "sha224", "sha256", "sha384", "sha512", "sha3_224", "sha3_256",
            "sha3_384", "sha3_512", "SHA3_256", "SHA3_384", "SHA3_512",
        ),
    )
    if setup is None:
        setup = _qualified_call_evidence(analyzer, "new", "hashlib.new")
    if setup is None and analyzer.language == "csharp":
        setup = _qualified_call_evidence(
            analyzer, "CreateHash", "IncrementalHash.CreateHash"
        )
    if setup is None and analyzer.language == "csharp":
        setup = csharp_digest_setup_evidence(analyzer)
    if setup is None and analyzer.language == "python":
        setup = _python_hash_callable_evidence(analyzer)

    update = _first_evidence(analyzer, ("update", "AppendData"))
    if update is None and analyzer.language == "csharp":
        update = _first_evidence(analyzer, ("TransformBlock",))
    if update is None:
        update = _hash_constructor_input_evidence(analyzer)
    if update is None and analyzer.language == "typescript":
        update = typescript_node_hash_pipeline_evidence(analyzer)

    terminal_names = (
        "digest", "hexdigest", "GetHashAndReset", "finalize", "HashData",
        "ComputeHash", "TransformFinalBlock",
    )
    if analyzer.language == "typescript":
        terminal_names += ("end", "read")
    terminal = (
        _python_hash_terminal_evidence(analyzer, terminal_names)
        if analyzer.language == "python"
        else _first_evidence(analyzer, terminal_names)
    )
    required: list[tuple[TaskEvidence | None, str]] = [
        (setup, "digest construction"),
        (terminal, "digest terminal operation"),
    ]
    if setup and setup.operation not in {"HashData", "ComputeHash"}:
        required.insert(1, (update, "digest input consumption"))
    return Analysis(source=context.source, required=tuple(required))


def validate_rule(
    context: TaskContext, analysis: Analysis
) -> TaskValidationResult | None:
    if context.language == "java":
        if analysis.chained_digest or analysis.digest_receivers:
            return None
        return java_task_result(
            context,
            success=False,
            failure_code="target_api_not_found",
            reason=(
                "F5 requires MessageDigest.getInstance assigned to a "
                "MessageDigest receiver"
            ),
        )
    matched = tuple(item for item, _ in analysis.required if item is not None)
    if matched:
        return None
    analyzer = require_analyzer(context)
    return TaskValidationResult(
        language=analyzer.language,
        task_id=context.task_id,
        success=False,
        reason_codes=("target_api_not_found",),
        reason="missing " + ", ".join(name for _, name in analysis.required),
    )


def validate_unkeyed_digest_rule(
    context: TaskContext,
    analysis: Analysis,
    result: TaskValidationResult,
) -> TaskValidationResult | None:
    if context.language == "java":
        return None
    analyzer = require_analyzer(context)
    if standalone_digest_operation(analyzer):
        return None
    return TaskValidationResult(
        language=context.language,
        task_id=context.task_id,
        success=False,
        reason_codes=("target_wrong_api_family",),
        reason=(
            "F5 requires a standalone unkeyed digest operation; a hash "
            "algorithm used only by HMAC is not a digest-family operation"
        ),
        matched_operations=result.matched_operations,
    )


# Compatibility name retained for callers outside the family package.
validate_target_misuse_rule = validate_unkeyed_digest_rule


def csharp_digest_setup_evidence(analyzer: SourceAnalyzer) -> TaskEvidence | None:
    for call in analyzer.calls("Create"):
        if re.fullmatch(
            r"(?:MD5|SHA1|SHA256|SHA384|SHA512|SHA3_256|SHA3_384|SHA3_512)\.Create",
            re.sub(r"\s+", "", call.callee),
        ):
            return _call_evidence(analyzer, call)
    return None


def standalone_digest_operation(analyzer: SourceAnalyzer) -> bool:
    if analyzer.language == "typescript":
        return bool(analyzer.calls("createHash"))
    if analyzer.language == "csharp":
        return any(
            "HMAC" not in re.sub(r"\s+", "", call.callee).upper()
            and "CREATEHMAC" not in re.sub(r"\s+", "", call.callee).upper()
            for call in analyzer.calls(
                "HashData", "ComputeHash", "Create", "CreateHash", "GetHashAndReset"
            )
        )
    if analyzer.language == "python":
        if analyzer.calls("Hash"):
            return True
        if any(
            call.callee.startswith("hashlib.")
            for call in analyzer.calls(
                "new", "md5", "sha1", "sha224", "sha256", "sha384", "sha512",
                "sha3_224", "sha3_256", "sha3_384", "sha3_512",
            )
        ):
            return True
        return any(
            _is_python_hashlib_callable(analyzer, call)
            or (_is_python_hash_callable(analyzer, call) and bool(call.arguments))
            for call in analyzer.calls()
            if call.simple_name not in {"HMAC", "new"}
        )
    return False


def typescript_node_hash_pipeline_evidence(
    analyzer: SourceAnalyzer,
) -> TaskEvidence | None:
    names = {
        re.sub(r"\s+", "", assignment.name).rsplit(".", 1)[-1]
        for assignment in getattr(analyzer, "assignments", ())
        if re.search(r"\bcreateHash\s*\(", assignment.expression)
    }
    for call in analyzer.calls("pipeline"):
        if not analyzer.position_is_locally_reachable(call.start) or len(call.arguments) < 2:
            continue
        for argument in call.arguments[1:]:
            compact = re.sub(r"\s+", "", argument)
            if compact in names or any(
                re.fullmatch(
                    rf"{re.escape(name)}as(?:any|unknown|NodeJS\.[A-Za-z_]\w*)",
                    compact,
                )
                for name in names
            ):
                return _call_evidence(analyzer, call)
    return None
