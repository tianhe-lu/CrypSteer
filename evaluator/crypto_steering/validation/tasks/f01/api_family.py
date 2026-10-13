"""F1 Rule 3: recognize a supported symmetric-cipher API family."""

from __future__ import annotations

from dataclasses import dataclass
import re

from crypto_steering.validation.java.source import strip_java_comments
from crypto_steering.validation.java.task_analysis import (
    _factory_receivers_assigned_from,
    _is_cipher_receiver,
    _receiver_method_calls,
)
from crypto_steering.validation.models import Language, TaskEvidence, TaskValidationResult
from crypto_steering.validation.source.index import analyzer_for
from crypto_steering.validation.tasks.shared.results import (
    java_task_result,
    require_analyzer,
    typescript_node_decryption_evidence,
)
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.evidence import (
    _cipher_terminal_evidence,
    _cipher_terminal_names,
    _first_evidence,
)


@dataclass(frozen=True)
class Analysis:
    """Evidence shared by the four F1 rule modules."""

    source: str
    setup: TaskEvidence | None = None
    terminal: TaskEvidence | None = None
    encrypt: TaskEvidence | None = None
    decrypt: TaskEvidence | None = None
    transformations: tuple[str, ...] = ()
    cipher_receivers: frozenset[str] = frozenset()
    candidate_initializations: tuple[tuple[str, object], ...] = ()


def analyze(context: TaskContext) -> Analysis:
    if context.language == "java":
        source = strip_java_comments(context.source)
        analyzer = analyzer_for("java", source)
        transformations = tuple(
            value
            for call in analyzer.calls("getInstance")
            if re.search(r"(?:^|\.)Cipher\s*\.\s*getInstance$", call.callee)
            and call.arguments
            for value in analyzer.resolve_strings(
                call.arguments[0], call.start
            ).values
        )
        receivers = frozenset(_factory_receivers_assigned_from(source, "Cipher"))
        initializations = tuple(
            (receiver, call)
            for receiver, call in _receiver_method_calls(source, "init")
            if receiver in receivers
            or _is_cipher_receiver(source, receiver, call.start)
        )
        return Analysis(
            source=source,
            transformations=transformations,
            cipher_receivers=receivers,
            candidate_initializations=initializations,
        )

    analyzer = require_analyzer(context)
    setup = _first_evidence(
        analyzer,
        (
            "createCipheriv", "createDecipheriv", "AesGcm", "AesCcm",
            "CreateEncryptor", "CreateDecryptor", "AESGCM", "AESCCM",
            "Fernet", "Cipher",
        ),
    )
    encrypt = _first_evidence(
        analyzer,
        ("createCipheriv", "CreateEncryptor", "encryptor", "encrypt",
         "Encrypt", "EncryptCbc"),
    )
    decrypt = _first_evidence(
        analyzer,
        ("createDecipheriv", "CreateDecryptor", "decryptor", "decrypt",
         "Decrypt", "DecryptCbc"),
    )
    return Analysis(
        source=context.source,
        setup=setup,
        terminal=_cipher_terminal_evidence(
            analyzer, _cipher_terminal_names(analyzer.language)
        ),
        encrypt=encrypt,
        decrypt=typescript_node_decryption_evidence(context, decrypt),
    )


def required_evidence(
    analysis: Analysis,
) -> tuple[tuple[TaskEvidence | None, str], ...]:
    """Return the family-wide minimum, independent of the T suffix."""

    direction = analysis.encrypt or analysis.decrypt
    return (
        (analysis.setup, "symmetric cipher setup"),
        (analysis.terminal, "completed symmetric cipher operation"),
        (direction, "encryption or decryption direction"),
    )


def validate_rule(
    context: TaskContext, analysis: Analysis
) -> TaskValidationResult | None:
    if context.language == "java":
        if analysis.transformations and all(
            value.strip().upper().startswith("RSA")
            for value in analysis.transformations
        ):
            return java_task_result(
                context,
                success=False,
                failure_code="target_wrong_api_family",
                reason="F1 requires a symmetric Cipher transformation rather than RSA",
            )
        if not analysis.cipher_receivers:
            return java_task_result(
                context,
                success=False,
                failure_code="target_api_not_found",
                reason="F1 requires Cipher.getInstance assigned to a Cipher receiver",
            )
        return None

    required = required_evidence(analysis)
    if any(item is not None for item, _ in required):
        return None
    analyzer = require_analyzer(context)
    return TaskValidationResult(
        language=analyzer.language,
        task_id=context.task_id,
        success=False,
        reason_codes=("target_api_not_found",),
        reason="missing " + ", ".join(name for _, name in required),
    )


def mode_sink_is_rewriteable(language: Language, source: str) -> bool:
    """Return whether a native M02 original exposes a controllable mode sink."""

    if language == "java":
        return bool(re.search(r"\bCipher\s*\.\s*getInstance\s*\(", source))
    if language == "typescript":
        return bool(re.search(r"\bcreate(?:Cipher|Decipher)iv\s*\(", source))
    if language == "csharp":
        return bool(
            re.search(r"\bAes\s*\.\s*Create\s*\(", source)
            or re.search(r"\bCipherMode\s*\.", source)
        )
    if language == "python":
        return bool(
            re.search(r"\bCipher\s*\(", source)
            and re.search(r"\bmodes\s*\.\s*[A-Za-z_]\w*\s*\(", source)
        )
    return False
