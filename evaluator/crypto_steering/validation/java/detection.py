"""Shared Java evidence assembly for category-owned detectors.

This module contains mechanics common to several Java detectors: locating
configured call arguments, applying source reachability, and converting a
category decision into the multilingual result schema.  It contains no M01-
M10 policy thresholds or secure/misuse classifications.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from crypto_steering.validation.detectors.core import (
    _line_column,
    _reason_code_for_label,
    _safe_id,
)
from crypto_steering.validation.java.calls import (
    constructor_calls,
    receiver_method_calls,
    receivers_assigned_from_get_instance,
    source_contains_any_token,
    static_method_calls,
)
from crypto_steering.validation.java.security_value_flow import SecurityValueFlow
from crypto_steering.validation.models import (
    DetectionResult,
    DetectorFinding,
    ProgramLabel,
    SourceSpan,
    combine_detector_findings,
)


class JavaAnalysis(Protocol):
    """The small analyzer surface needed to assemble Java findings."""

    source: str
    flow: SecurityValueFlow

    def position_is_reachable(self, index: int) -> bool: ...


@dataclass(frozen=True)
class JavaSinkSpec:
    """One Java call argument that a category treats as policy evidence."""

    id: str
    kind: str
    sink_name: str
    class_name: str = ""
    method_name: str = ""
    argument_index: int | None = None
    receiver_factory_class: str = ""


@dataclass(frozen=True)
class JavaPolicyArgument:
    spec: JavaSinkSpec
    expression: str
    start: int
    end: int
    call_start: int
    call_end: int
    receiver: str | None = None
    receiver_algorithm: str | None = None


@dataclass(frozen=True)
class JavaEvidence:
    """A category classification before common result conversion."""

    sink_name: str
    expression: str
    label: ProgramLabel
    reason: str
    start: int = -1
    end: int = -1
    sink_id: str | None = None
    reason_code: str | None = None
    policy_id: str | None = None


def iter_policy_arguments(
    source: str,
    specs: Iterable[JavaSinkSpec],
) -> Iterable[JavaPolicyArgument]:
    """Yield configured arguments in stable policy and source order."""

    items = tuple(specs)
    receiver_algorithms: dict[str, str] = {}
    for factory_class in {
        spec.receiver_factory_class
        for spec in items
        if spec.receiver_factory_class
    }:
        receiver_algorithms.update(
            receivers_assigned_from_get_instance(source, factory_class)
        )

    for spec in items:
        argument_index = spec.argument_index
        if argument_index is None:
            continue
        if spec.kind == "static_method":
            calls = (
                (None, call)
                for call in static_method_calls(
                    source,
                    spec.class_name,
                    spec.method_name,
                )
            )
        elif spec.kind == "constructor":
            calls = (
                (None, call)
                for call in constructor_calls(source, spec.class_name)
            )
        elif spec.kind == "receiver_method":
            calls = receiver_method_calls(source, spec.method_name)
        else:
            continue

        for receiver, call in calls:
            receiver_algorithm = (
                receiver_algorithms.get(receiver)
                if receiver is not None
                else None
            )
            if (
                spec.kind == "receiver_method"
                and receiver_algorithm is None
            ):
                continue
            if len(call.arguments) <= argument_index:
                continue
            argument = call.arguments[argument_index]
            yield JavaPolicyArgument(
                spec=spec,
                expression=argument.text,
                start=argument.start,
                end=argument.end,
                call_start=call.start,
                call_end=call.end,
                receiver=receiver,
                receiver_algorithm=receiver_algorithm,
            )


def source_may_contain_policy_sink(
    source: str,
    specs: Iterable[JavaSinkSpec],
    *,
    extra_tokens: Iterable[str] = (),
) -> bool:
    """Conservatively skip Java parsing when no registered sink can occur."""

    tokens = {token for token in extra_tokens if token}
    for spec in specs:
        tokens.update(
            token
            for token in (
                spec.class_name,
                spec.receiver_factory_class,
                (
                    spec.method_name
                    if not spec.class_name
                    and not spec.receiver_factory_class
                    else ""
                ),
            )
            if token
        )
    return source_contains_any_token(source, tokens)


def evidence_result(
    analyzer: JavaAnalysis,
    misuse_id: str,
    evidence: Iterable[JavaEvidence],
    *,
    no_sink_reason: str,
    preserve_filtered_summary: bool = True,
    apply_reachability: bool = True,
) -> DetectionResult:
    """Apply Java reachability and build the common detector result exactly."""

    items = tuple(evidence)
    findings = [
        finding
        for item in items
        if (
            finding := finding_from_evidence(
                analyzer,
                misuse_id,
                item,
                apply_reachability=apply_reachability,
            )
        )
        is not None
    ]
    effective_no_sink_reason = no_sink_reason
    if preserve_filtered_summary and items:
        effective_no_sink_reason = _legacy_evidence_summary(items)
    return combine_detector_findings(
        misuse_id,
        "java",
        findings,
        no_sink_reason=effective_no_sink_reason,
    )


def finding_from_evidence(
    analyzer: JavaAnalysis,
    misuse_id: str,
    evidence: JavaEvidence,
    *,
    apply_reachability: bool = True,
) -> DetectorFinding | None:
    """Convert one reachable Java category decision into a shared finding."""

    if (
        apply_reachability
        and evidence.start >= 0
        and not analyzer.position_is_reachable(evidence.start)
    ):
        return None
    line, column = _line_column(analyzer.source, evidence.start)
    return DetectorFinding(
        misuse_id=misuse_id,
        language="java",
        sink_id=(
            evidence.sink_id
            or _safe_id(evidence.sink_name or "java_sink")
        ),
        sink_name=evidence.sink_name or "Java security sink",
        expression=evidence.expression,
        label=evidence.label,
        reason_code=(
            evidence.reason_code
            or _reason_code_for_label(evidence.label)
        ),
        reason=evidence.reason,
        policy_id=evidence.policy_id or f"{misuse_id}.policy.v1",
        span=SourceSpan(
            start=evidence.start,
            end=evidence.end,
            line=line,
            column=column,
        ),
    )


def parse_failure_result(
    analyzer: JavaAnalysis,
    misuse_id: str,
) -> DetectionResult:
    """Preserve the Java parser failure contract for a category."""

    return evidence_result(
        analyzer,
        misuse_id,
        (),
        no_sink_reason=(
            "javalang parse failed: "
            f"{analyzer.flow.ast_context.parse_error}"
        ),
    )


def _legacy_evidence_summary(evidence: tuple[JavaEvidence, ...]) -> str:
    """Retain the pre-reachability Java summary used by saved results."""

    labels = {item.label for item in evidence}
    if "mixed" in labels or {"secure", "misuse"} <= labels:
        return "both secure and misuse sinks were found"
    if "misuse" in labels:
        return "at least one sink uses a hardcoded or predictable source"
    if labels == {"secure"}:
        return "all sinks use non-constant sources"
    return "sinks found, but sources were unresolved"


__all__ = (
    "JavaEvidence",
    "JavaPolicyArgument",
    "JavaSinkSpec",
    "evidence_result",
    "finding_from_evidence",
    "iter_policy_arguments",
    "parse_failure_result",
    "source_may_contain_policy_sink",
)
