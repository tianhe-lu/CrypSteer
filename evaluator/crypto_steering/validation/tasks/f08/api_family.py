"""F8 Rule 3: recognize protected-key storage API families."""

from __future__ import annotations

from dataclasses import dataclass

from crypto_steering.validation.java.source import strip_java_comments
from crypto_steering.validation.java.static_analysis import actual_symbol_usage, extract_assignments
from crypto_steering.validation.java.task_analysis import _keystore_operation_calls
from crypto_steering.validation.models import TaskEvidence, TaskValidationResult
from crypto_steering.validation.tasks.shared.results import java_task_result, require_analyzer
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.evidence import _first_evidence


@dataclass(frozen=True)
class Analysis:
    source: str
    required: tuple[tuple[TaskEvidence | None, str], ...] = ()
    keystore_used: bool = False
    assignments: tuple[object, ...] = ()
    keystore_operations: tuple[object, ...] = ()


def analyze(context: TaskContext) -> Analysis:
    if context.language == "java":
        source = strip_java_comments(context.source)
        assignments = tuple(extract_assignments(source))
        return Analysis(
            source=source,
            keystore_used=actual_symbol_usage(source, "KeyStore"),
            assignments=assignments,
            keystore_operations=tuple(
                _keystore_operation_calls(source, list(assignments))
            ),
        )

    analyzer = require_analyzer(context)
    native_terminal = _first_evidence(
        analyzer,
        (
            "serialize_key_and_certificates",
            "ExportEncryptedPkcs8PrivateKey",
            "TryExportEncryptedPkcs8PrivateKey",
        ),
    )
    if native_terminal is not None:
        output = _first_evidence(
            analyzer,
            ("writeFile", "writeFileSync", "WriteAllBytes", "write_bytes", "write", "open"),
        )
        return Analysis(
            source=context.source,
            required=(
                (native_terminal, "native protected serialization"),
                (output, "protected output write"),
            ),
        )

    kdf = _first_evidence(
        analyzer,
        ("pbkdf2", "pbkdf2Sync", "pbkdf2_hmac", "Pbkdf2", "Rfc2898DeriveBytes", "PBKDF2HMAC"),
    )
    aead_setup = _first_evidence(analyzer, ("createCipheriv", "AesGcm", "AESGCM"))
    if analyzer.language == "typescript":
        terminal = _first_evidence(analyzer, ("final",))
        tag = _first_evidence(analyzer, ("getAuthTag",))
    elif analyzer.language == "csharp":
        terminal = _first_evidence(analyzer, ("Encrypt",))
        tag = terminal
    else:
        terminal = _first_evidence(analyzer, ("encrypt",))
        tag = terminal
    output = _first_evidence(
        analyzer, ("writeFile", "writeFileSync", "WriteAllBytes", "write_bytes", "write")
    )
    return Analysis(
        source=context.source,
        required=(
            (kdf, "password-based key derivation"),
            (aead_setup, "authenticated-encryption construction"),
            (terminal, "completed authenticated encryption"),
            (tag, "authentication tag retention"),
            (output, "protected-entry file write"),
        ),
    )


def validate_rule(
    context: TaskContext, analysis: Analysis
) -> TaskValidationResult | None:
    if context.language == "java":
        if analysis.keystore_used:
            return None
        return java_task_result(
            context,
            success=False,
            failure_code="target_api_not_found",
            reason="F8 requires actual KeyStore usage",
        )
    if any(item is not None for item, _ in analysis.required):
        return None
    return TaskValidationResult(
        language=context.language,
        task_id=context.task_id,
        success=False,
        reason_codes=("target_api_not_found",),
        reason="missing " + ", ".join(name for _, name in analysis.required),
    )
