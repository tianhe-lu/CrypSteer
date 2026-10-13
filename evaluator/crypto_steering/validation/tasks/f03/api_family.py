"""F3 Rule 3: recognize supported asymmetric-encryption APIs."""

from __future__ import annotations

from dataclasses import dataclass
import re

from crypto_steering.validation.java.source import strip_java_comments
from crypto_steering.validation.java.task_analysis import (
    _cipher_receivers_with_transformation,
    _receiver_method_calls,
)
from crypto_steering.validation.models import TaskEvidence, TaskValidationResult
from crypto_steering.validation.source.index import analyzer_for
from crypto_steering.validation.tasks.shared.results import java_task_result, require_analyzer
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.evidence import (
    _first_evidence,
    _rsa_setup_evidence,
)


@dataclass(frozen=True)
class Analysis:
    source: str
    setup: TaskEvidence | None = None
    encrypt: TaskEvidence | None = None
    decrypt: TaskEvidence | None = None
    rsa_receivers: frozenset[str] = frozenset()
    rsa_transformations: tuple[str, ...] = ()
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
            if value.strip().upper().startswith("RSA")
        )
        receivers = frozenset(
            _cipher_receivers_with_transformation(source, "RSA")
        )
        return Analysis(
            source=source,
            rsa_receivers=receivers,
            rsa_transformations=transformations,
            candidate_initializations=tuple(
                (receiver, call)
                for receiver, call in _receiver_method_calls(source, "init")
                if receiver in receivers
            ),
        )
    analyzer = require_analyzer(context)
    encrypt = _first_evidence(
        analyzer, ("publicEncrypt", "Encrypt", "encrypt")
    )
    decrypt = _first_evidence(
        analyzer, ("privateDecrypt", "Decrypt", "decrypt")
    )
    return Analysis(
        source=context.source,
        # A key supplied by the caller is valid key material. In that case the
        # operation call itself is the available static evidence; local key
        # generation or deserialization is not a family-wide requirement.
        setup=_rsa_setup_evidence(analyzer) or encrypt or decrypt,
        encrypt=encrypt,
        decrypt=decrypt,
    )


def required_evidence(
    analysis: Analysis,
) -> tuple[tuple[TaskEvidence | None, str], ...]:
    return (
        (analysis.setup, "asymmetric key generation/material"),
        (analysis.encrypt or analysis.decrypt, "asymmetric encryption or decryption"),
    )


def validate_rule(
    context: TaskContext, analysis: Analysis
) -> TaskValidationResult | None:
    if context.language == "java":
        if analysis.rsa_receivers:
            return None
        return java_task_result(
            context,
            success=False,
            failure_code="target_api_not_found",
            reason="F3 requires Cipher.getInstance with an RSA transformation",
        )
    required = required_evidence(analysis)
    matched = tuple(item for item, _ in required if item is not None)
    if matched:
        return None
    analyzer = require_analyzer(context)
    return TaskValidationResult(
        language=analyzer.language,
        task_id=context.task_id,
        success=False,
        reason_codes=("target_api_not_found",),
        reason="missing " + ", ".join(name for _, name in required),
    )
