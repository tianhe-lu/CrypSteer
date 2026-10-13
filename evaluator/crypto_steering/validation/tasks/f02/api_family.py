"""F2 Rule 3: recognize CBC-capable symmetric-cipher APIs."""

from __future__ import annotations

from dataclasses import dataclass

from crypto_steering.validation.java.source import strip_java_comments
from crypto_steering.validation.java.task_analysis import (
    _cipher_receivers_with_transformation,
    _is_cipher_receiver,
    _receiver_method_calls,
)
from crypto_steering.validation.models import TaskEvidence, TaskValidationResult
from crypto_steering.validation.tasks.shared.results import (
    java_task_result,
    require_analyzer,
    typescript_node_decryption_evidence,
)
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.evidence import (
    _cbc_evidence,
    _cipher_terminal_evidence,
    _cipher_terminal_names,
    _first_evidence,
)


@dataclass(frozen=True)
class Analysis:
    source: str
    setup: TaskEvidence | None = None
    terminal: TaskEvidence | None = None
    encrypt: TaskEvidence | None = None
    decrypt: TaskEvidence | None = None
    explicitly_non_cbc: bool = False
    cipher_receivers: frozenset[str] = frozenset()
    candidate_initializations: tuple[tuple[str, object], ...] = ()


def analyze(context: TaskContext) -> Analysis:
    if context.language == "java":
        from crypto_steering.validation.tasks.f02.sequence import (
            encrypt_path_is_explicitly_non_cbc,
        )

        source = strip_java_comments(context.source)
        receivers = frozenset(_cipher_receivers_with_transformation(source, "CBC"))
        initializations = tuple(
            (receiver, call)
            for receiver, call in _receiver_method_calls(source, "init")
            if receiver in receivers
            or _is_cipher_receiver(source, receiver, call.start)
        )
        return Analysis(
            source=source,
            explicitly_non_cbc=encrypt_path_is_explicitly_non_cbc(source),
            cipher_receivers=receivers,
            candidate_initializations=initializations,
        )

    analyzer = require_analyzer(context)
    decrypt = _first_evidence(
        analyzer,
        ("createDecipheriv", "CreateDecryptor", "decryptor", "decrypt",
         "Decrypt", "DecryptCbc"),
    )
    return Analysis(
        source=context.source,
        setup=_cbc_evidence(analyzer),
        terminal=_cipher_terminal_evidence(
            analyzer, _cipher_terminal_names(analyzer.language)
        ),
        encrypt=_first_evidence(
            analyzer,
            ("createCipheriv", "CreateEncryptor", "encryptor", "encrypt",
             "Encrypt", "EncryptCbc"),
        ),
        decrypt=typescript_node_decryption_evidence(context, decrypt),
    )


def required_evidence(
    analysis: Analysis,
) -> tuple[tuple[TaskEvidence | None, str], ...]:
    """Return the CBC family minimum, independent of the T suffix."""

    direction = analysis.encrypt or analysis.decrypt
    return (
        (analysis.setup, "CBC mode with an IV"),
        (analysis.terminal, "completed CBC operation"),
        (direction, "CBC encryption or decryption direction"),
    )


def validate_rule(
    context: TaskContext, analysis: Analysis
) -> TaskValidationResult | None:
    if context.language == "java":
        if not analysis.cipher_receivers:
            return java_task_result(
                context,
                success=False,
                failure_code="target_api_not_found",
                reason="F2 requires Cipher.getInstance with CBC transformation",
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
