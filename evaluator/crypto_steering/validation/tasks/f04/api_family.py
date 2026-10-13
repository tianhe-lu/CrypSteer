"""F4 Rule 3: recognize supported digital-signature APIs."""

from __future__ import annotations

from dataclasses import dataclass

from crypto_steering.validation.java.source import strip_java_comments
from crypto_steering.validation.java.task_analysis import (
    _factory_receivers_assigned_from,
    _receiver_method_calls,
)
from crypto_steering.validation.models import TaskEvidence, TaskValidationResult
from crypto_steering.validation.source.index import SourceAnalyzer
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
    sign: TaskEvidence | None = None
    verify: TaskEvidence | None = None
    signature_receivers: frozenset[str] = frozenset()
    calls_by_receiver: tuple[tuple[str, tuple[tuple[int, str, object], ...]], ...] = ()


def analyze(context: TaskContext) -> Analysis:
    if context.language == "java":
        source = strip_java_comments(context.source)
        receivers = frozenset(
            _factory_receivers_assigned_from(source, "Signature")
        )
        calls: dict[str, list[tuple[int, str, object]]] = {
            receiver: [] for receiver in receivers
        }
        for method_name in ("initSign", "initVerify", "update", "sign", "verify"):
            for receiver, call in _receiver_method_calls(source, method_name):
                if receiver in receivers:
                    calls[receiver].append((call.start, method_name, call))
        return Analysis(
            source=source,
            signature_receivers=receivers,
            calls_by_receiver=tuple(
                (receiver, tuple(items)) for receiver, items in calls.items()
            ),
        )

    analyzer = require_analyzer(context)
    if context.language == "python":
        sign = _python_signature_evidence(analyzer, "sign")
        verify = _python_signature_evidence(analyzer, "verify")
    else:
        sign = _first_evidence(analyzer, ("sign", "SignData", "SignHash"))
        verify = _first_evidence(analyzer, ("verify", "VerifyData", "VerifyHash"))
    signature_setup = _first_evidence(analyzer, ("createSign", "createVerify"))
    return Analysis(
        source=context.source,
        # A caller-supplied key is acceptable, and Node's streaming signature
        # object is itself family evidence. Rule 4 still requires sign/verify.
        setup=_rsa_setup_evidence(analyzer) or signature_setup or sign or verify,
        sign=sign,
        verify=verify,
    )


def _python_signature_evidence(
    analyzer: SourceAnalyzer,
    operation: str,
) -> TaskEvidence | None:
    """Return native Python signature evidence, excluding wrapper calls.

    A bare ``sign(...)`` helper call is not itself a cryptographic API, and
    ``x509.CertificateBuilder.sign`` signs a certificate rather than the
    canonical message.  Dotted calls on key or signature objects remain
    eligible so PyCA, PyCryptodome, and equivalent approved APIs are covered.
    """

    for call in analyzer.calls(operation):
        if "." not in call.callee or not analyzer.position_is_locally_reachable(
            call.start
        ):
            continue
        if operation == "verify" and not call.arguments:
            # PyCA verification is a direct call whose first argument is the
            # signature.  ``key.verify()`` is an obsolete context-factory
            # shape, not a completed verification operation.
            continue
        expression = analyzer.source[call.start : call.end]
        if operation == "sign" and "CertificateBuilder" in expression:
            continue
        return TaskEvidence(
            operation=call.simple_name,
            expression=expression,
            span=analyzer.source_span(call.start, call.end),
        )
    return None


def required_evidence(
    analysis: Analysis,
) -> tuple[tuple[TaskEvidence | None, str], ...]:
    return (
        (analysis.setup, "asymmetric signature key/API setup"),
        (analysis.sign or analysis.verify, "signature generation or verification"),
    )


def validate_rule(
    context: TaskContext, analysis: Analysis
) -> TaskValidationResult | None:
    if context.language == "java":
        if analysis.signature_receivers:
            return None
        return java_task_result(
            context,
            success=False,
            failure_code="target_api_not_found",
            reason="F4 requires Signature.getInstance assigned to a Signature receiver",
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
