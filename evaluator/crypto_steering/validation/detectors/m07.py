"""M07 KDF-iteration detector."""

from __future__ import annotations

import re

from crypto_steering.validation.java.detection import (
    JavaEvidence,
    JavaSinkSpec,
    evidence_result,
    iter_policy_arguments,
    parse_failure_result,
    source_may_contain_policy_sink,
)
from crypto_steering.validation.models import (
    DetectionResult,
    DetectorFinding,
    Language,
)
from crypto_steering.validation.source.index import (
    CStyleAnalyzer,
    CallExpression,
    JavaAnalyzer,
    SourceAnalyzer,
)
from crypto_steering.validation.detectors.apis import (
    _is_csharp_framework_pbkdf2_call,
    _typescript_kdf_calls,
    _is_python_hashlib_pbkdf2_call,
)
from crypto_steering.validation.detectors.core import (
    _iteration_finding,
    _result,
    _suppress_unresolved_uncalled_helpers,
    _suppress_proven_inactive_findings,
    _preserve_definite_inactive_findings,
    _missing_argument_finding,
    _call_argument,
)
from crypto_steering.validation.detectors.engine import execute_detector


_JAVA_SINKS = (
    JavaSinkSpec(
        id="pbe_key_spec_iterations",
        kind="constructor",
        class_name="PBEKeySpec",
        sink_name="new PBEKeySpec iterations",
        argument_index=2,
    ),
    JavaSinkSpec(
        id="pbe_parameter_spec_iterations",
        kind="constructor",
        class_name="PBEParameterSpec",
        sink_name="new PBEParameterSpec iterations",
        argument_index=1,
    ),
)
_JAVA_MINIMUM_ITERATIONS = 1000


def _detect_m07(analyzer: SourceAnalyzer, task_id: str | None, modern: int) -> DetectionResult:
    if isinstance(analyzer, JavaAnalyzer):
        return _detect_java_m07(analyzer, task_id)
    del task_id
    findings: list[DetectorFinding] = []
    call_specs: tuple[tuple[str, int], ...]
    if analyzer.language == "typescript":
        call_specs = ()
    elif analyzer.language == "csharp":
        call_specs = (("Pbkdf2", 2), ("Rfc2898DeriveBytes", 2))
    else:
        call_specs = ()
    for name, argument_index in call_specs:
        for call in analyzer.calls(name):
            if analyzer.language == "csharp":
                if not _is_csharp_framework_pbkdf2_call(analyzer, call):
                    continue
                expression = _csharp_effective_pbkdf2_iterations(analyzer, call)
            else:
                expression = _call_argument(
                    call,
                    argument_index,
                    *(('iterations', 'iterationCount') if analyzer.language == "csharp" else ()),
                )
            if expression is None:
                findings.append(_missing_argument_finding(analyzer, "M07", call, "kdf_iterations"))
            else:
                findings.append(_iteration_finding(analyzer, call, expression, modern))
    if analyzer.language == "typescript":
        for call, _canonical in _typescript_kdf_calls(
            analyzer,
            ("pbkdf2", "pbkdf2Sync"),
        ):
            expression = _call_argument(call, 2, "iterations")
            if expression is None:
                findings.append(
                    _missing_argument_finding(
                        analyzer,
                        "M07",
                        call,
                        "kdf_iterations",
                    )
                )
            else:
                findings.append(
                    _iteration_finding(analyzer, call, expression, modern)
                )
    if analyzer.language == "python":
        for call in analyzer.calls("PBKDF2HMAC"):
            expression = call.named_arguments.get("iterations")
            if expression is None:
                findings.append(_missing_argument_finding(analyzer, "M07", call, "kdf_iterations"))
            else:
                findings.append(_iteration_finding(analyzer, call, expression, modern))
        for call in analyzer.calls("pbkdf2_hmac"):
            if not _is_python_hashlib_pbkdf2_call(analyzer, call):
                continue
            expression = _call_argument(call, 3, "iterations")
            if expression is None:
                findings.append(_missing_argument_finding(analyzer, "M07", call, "kdf_iterations"))
            else:
                findings.append(_iteration_finding(analyzer, call, expression, modern))
    lexical_findings = tuple(findings)
    findings = _suppress_proven_inactive_findings(analyzer, findings)
    findings = _suppress_unresolved_uncalled_helpers(analyzer, findings)
    findings = _preserve_definite_inactive_findings(
        analyzer,
        lexical_findings,
        findings,
    )
    return _result(analyzer, "M07", findings, "no PBKDF2 iteration sink found")


def _detect_java_m07(
    analyzer: JavaAnalyzer,
    task_id: str | None,
) -> DetectionResult:
    del task_id
    if not source_may_contain_policy_sink(analyzer.source, _JAVA_SINKS):
        return evidence_result(
            analyzer,
            "M07",
            (),
            no_sink_reason="no KDF iteration sink found",
        )
    if not analyzer.flow.ast_context.parse_success:
        return parse_failure_result(analyzer, "M07")

    evidence: list[JavaEvidence] = []
    for argument in iter_policy_arguments(analyzer.source, _JAVA_SINKS):
        value = analyzer.int_value(argument.expression, argument.start)
        if value is None:
            label = "unknown"
            reason = (
                f"{argument.spec.sink_name} value could not be resolved"
            )
        elif value < _JAVA_MINIMUM_ITERATIONS:
            label = "misuse"
            reason = (
                f"iteration count {value} is below minimum "
                f"{_JAVA_MINIMUM_ITERATIONS}"
            )
        else:
            label = "secure"
            reason = (
                f"iteration count {value} meets minimum "
                f"{_JAVA_MINIMUM_ITERATIONS}"
            )
        evidence.append(
            JavaEvidence(
                sink_name=argument.spec.sink_name,
                expression=argument.expression,
                label=label,
                reason=reason,
                start=argument.start,
                end=argument.end,
            )
        )
    return evidence_result(
        analyzer,
        "M07",
        evidence,
        no_sink_reason="no KDF iteration sink found",
    )


def _csharp_effective_pbkdf2_iterations(
    analyzer: SourceAnalyzer,
    call: CallExpression,
) -> str | None:
    named = next(
        (
            call.named_arguments[name]
            for name in ("iterations", "iterationCount")
            if name in call.named_arguments
        ),
        None,
    )
    explicit = named
    if explicit is None and len(call.arguments) > 2:
        third = call.arguments[2]
        third_assignment = analyzer.latest_assignment(third, call.start)
        third_is_destination = bool(
            third_assignment is not None
            and re.fullmatch(
                r"\s*(?:new\s+byte\s*\[[^]]+\]|"
                r"stackalloc\s+byte\s*\[[^]]+\]|"
                r"[A-Za-z_][A-Za-z0-9_.]*\s*\.\s*AsSpan\s*\([^)]*\))\s*",
                third_assignment.expression,
                flags=re.IGNORECASE,
            )
        )
        third_values = analyzer.resolve_ints(third, call.start)
        if (
            call.simple_name == "Pbkdf2"
            and len(call.arguments) > 3
            and third_is_destination
        ):
            explicit = call.arguments[3]
        elif third_values.values or (
            len(call.arguments) > 3
            and "HashAlgorithm" in call.arguments[3]
        ):
            explicit = third
        elif call.simple_name == "Pbkdf2" and len(call.arguments) > 3:
            # Span-based static overload:
            # Pbkdf2(password, salt, destination, iterations, hashAlgorithm)
            explicit = call.arguments[3]
    if not isinstance(analyzer, CStyleAnalyzer):
        return explicit

    statement_start = max(
        analyzer.source.rfind(";", 0, call.start),
        analyzer.source.rfind("{", 0, call.start),
        analyzer.source.rfind("}", 0, call.start),
        analyzer.source.rfind("\n", 0, call.start),
    ) + 1
    prefix = analyzer.source[statement_start:call.start]
    receiver_match = re.search(
        r"\b(?P<receiver>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:new\s*)?$",
        prefix,
    )
    if receiver_match is not None:
        receiver = receiver_match.group("receiver")
        override = re.search(
            rf"\b{re.escape(receiver)}\s*\.\s*IterationCount\s*=\s*"
            r"(?P<expression>[^;]+);",
            analyzer.source[call.end :],
        )
        if override is not None:
            return override.group("expression").strip()

    if explicit is not None:
        return explicit
    # Legacy Rfc2898DeriveBytes overloads use 1,000 iterations. The benchmark
    # records this effective value while also reporting modern-guidance status.
    return "1000"



MISUSE_ID = "M07"


def detect(
    language: Language,
    source: str,
    *,
    task_id: str | None,
    modern_kdf_iterations: int,
    analyzer: SourceAnalyzer | None,
    analyzer_error: Exception | None = None,
) -> DetectionResult:
    return execute_detector(
        language,
        MISUSE_ID,
        source,
        task_id=task_id,
        modern_kdf_iterations=modern_kdf_iterations,
        analyzer=analyzer,
        analyzer_error=analyzer_error,
        implementation=_detect_m07,
    )


__all__ = ("MISUSE_ID", "detect")
