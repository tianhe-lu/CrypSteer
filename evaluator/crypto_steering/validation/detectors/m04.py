"""M04 key-size detector."""

from __future__ import annotations

import re

from crypto_steering.validation.java.detection import (
    JavaEvidence,
    JavaPolicyArgument,
    JavaSinkSpec,
    evidence_result,
    iter_policy_arguments,
    parse_failure_result,
    source_may_contain_policy_sink,
)
from crypto_steering.validation.java.resolution import normalize_algorithm
from crypto_steering.validation.models import (
    DetectionResult,
    DetectorFinding,
    Language,
    ProgramLabel,
)
from crypto_steering.validation.source.index import (
    CStyleAnalyzer,
    CallExpression,
    JavaAnalyzer,
    SourceAnalyzer,
)
from crypto_steering.validation.detectors.apis import (
    _is_typescript_crypto_api_call,
    _is_python_rsa_generate_private_key_call,
    _has_csharp_rsa_terminal,
    _is_csharp_rsa_key_size_assignment,
)
from crypto_steering.validation.detectors.core import (
    _algorithm_finding,
    _integer_policy_finding,
    _result,
    _suppress_proven_inactive_findings,
    _missing_argument_finding,
    _direct_call_finding,
    _object_property,
    _call_argument,
    _argument_value,
    _normalize_algorithm,
)
from crypto_steering.validation.detectors.engine import execute_detector


_JAVA_SINKS = (
    JavaSinkSpec(
        id="key_pair_generator_initialize_size",
        kind="receiver_method",
        receiver_factory_class="KeyPairGenerator",
        method_name="initialize",
        sink_name="KeyPairGenerator.initialize key size",
        argument_index=0,
    ),
    JavaSinkSpec(
        id="rsa_key_gen_parameter_spec_size",
        kind="constructor",
        class_name="RSAKeyGenParameterSpec",
        sink_name="new RSAKeyGenParameterSpec key size",
        argument_index=0,
    ),
    JavaSinkSpec(
        id="ec_gen_parameter_spec_curve",
        kind="constructor",
        class_name="ECGenParameterSpec",
        sink_name="new ECGenParameterSpec curve",
        argument_index=0,
    ),
)
_JAVA_MINIMUM_KEY_SIZES = {
    "RSA": 2048,
    "DSA": 2048,
    "DH": 2048,
    "EC": 224,
}
_JAVA_WEAK_EC_CURVES = {
    "SECP112R1",
    "SECP112R2",
    "SECP128R1",
    "SECP128R2",
    "SECP160K1",
    "SECP160R1",
    "SECP160R2",
    "SECP192K1",
    "SECP192R1",
    "PRIME192V1",
}

_NODE_FIXED_STRENGTH_KEY_TYPES = {
    "ED25519",
    "ED448",
    "X25519",
    "X448",
}
_NODE_WEAK_EC_CURVES = {
    "SECP112R1",
    "SECP112R2",
    "SECP128R1",
    "SECP128R2",
    "SECP160K1",
    "SECP160R1",
    "SECP160R2",
    "SECP192K1",
    "SECP192R1",
    "PRIME192V1",
}


def _detect_m04(analyzer: SourceAnalyzer, task_id: str | None, modern: int) -> DetectionResult:
    if isinstance(analyzer, JavaAnalyzer):
        return _detect_java_m04(analyzer, task_id)
    del modern
    findings: list[DetectorFinding] = []
    if analyzer.language == "typescript":
        for call in analyzer.calls("generateKeyPair", "generateKeyPairSync"):
            if not _is_typescript_crypto_api_call(analyzer, call):
                continue
            key_type = (
                analyzer.resolve_strings(call.arguments[0], call.start)
                if call.arguments
                else None
            )
            normalized_types = (
                {
                    _normalize_algorithm(value)
                    for value in key_type.values
                }
                if key_type is not None
                else set()
            )
            options = (
                _argument_value(call.arguments[1])
                if len(call.arguments) > 1
                else ""
            )
            if (
                normalized_types
                and normalized_types <= _NODE_FIXED_STRENGTH_KEY_TYPES
            ):
                findings.append(
                    _direct_call_finding(
                        analyzer,
                        "M04",
                        call,
                        "secure",
                        "approved_fixed_strength_key_type",
                        "the selected key type has approved fixed strength",
                    )
                )
                continue
            if normalized_types == {"EC"}:
                curve = _object_property(options, "namedCurve") if options else None
                if curve is None:
                    findings.append(
                        _missing_argument_finding(
                            analyzer,
                            "M04",
                            call,
                            "ec_named_curve",
                        )
                    )
                else:
                    findings.append(
                        _algorithm_finding(
                            analyzer,
                            "M04",
                            call,
                            curve,
                            analyzer.resolve_strings(curve, call.start),
                            _classify_node_curve,
                        )
                    )
                continue
            expression = call.named_arguments.get("modulusLength")
            if expression is None and len(call.arguments) > 1:
                expression = _object_property(options, "modulusLength")
                if expression is None and re.fullmatch(
                    r"[A-Za-z_$][A-Za-z0-9_$]*(?:\s*\.\s*[A-Za-z_$][A-Za-z0-9_$]*)*",
                    options,
                ):
                    expression = f"{options}.modulusLength"
            if expression is None:
                findings.append(_missing_argument_finding(analyzer, "M04", call, "rsa_modulus_length"))
            else:
                findings.append(_integer_policy_finding(analyzer, "M04", call, expression, 2048, "rsa_key_size"))
    elif analyzer.language == "csharp":
        rsa_create_calls = tuple(
            call
            for call in analyzer.calls("Create")
            if re.search(
                r"(?:^|\.)RSA\s*\.\s*Create$",
                re.sub(r"\s+", "", call.callee),
            )
        )
        rsa_provider_calls = tuple(
            analyzer.calls("RSACryptoServiceProvider")
        )
        rsa_key_size_assignments = (
            tuple(
                assignment
                for assignment in analyzer.assignments
                if _is_csharp_rsa_key_size_assignment(
                    analyzer,
                    assignment.name,
                    task_id,
                )
            )
            if isinstance(analyzer, CStyleAnalyzer)
            else ()
        )
        has_explicit_key_size = bool(rsa_key_size_assignments) or any(
            _call_argument(call, 0, "keySize") is not None
            for call in rsa_create_calls
        )
        for call in rsa_create_calls:
            expression = _call_argument(call, 0, "keySize")
            if expression is not None:
                findings.append(_integer_policy_finding(analyzer, "M04", call, expression, 2048, "rsa_key_size"))
            elif not has_explicit_key_size and _has_csharp_rsa_terminal(analyzer.source):
                findings.append(_direct_call_finding(analyzer, "M04", call, "secure", "secure_platform_default", "RSA.Create() uses a platform default of at least 2048 bits"))
        for call in rsa_provider_calls:
            expression = _call_argument(call, 0, "dwKeySize", "keySize")
            if expression is None:
                findings.append(
                    _missing_argument_finding(
                        analyzer,
                        "M04",
                        call,
                        "rsa_key_size",
                    )
                )
            else:
                findings.append(
                    _integer_policy_finding(
                        analyzer,
                        "M04",
                        call,
                        expression,
                        2048,
                        "rsa_key_size",
                    )
                )
        for assignment in rsa_key_size_assignments:
            synthetic = CallExpression("RSA.KeySize", (assignment.expression,), {}, assignment.start, assignment.end)
            findings.append(_integer_policy_finding(analyzer, "M04", synthetic, assignment.expression, 2048, "rsa_key_size"))
    elif analyzer.language == "python":
        for call in analyzer.calls("generate_private_key"):
            if not _is_python_rsa_generate_private_key_call(analyzer, call):
                continue
            expression = _call_argument(call, 1, "key_size")
            if expression is None:
                findings.append(_missing_argument_finding(analyzer, "M04", call, "rsa_key_size"))
            else:
                findings.append(_integer_policy_finding(analyzer, "M04", call, expression, 2048, "rsa_key_size"))
    findings = _suppress_proven_inactive_findings(analyzer, findings)
    return _result(analyzer, "M04", findings, "no RSA key-generation size sink found")


def _classify_node_curve(value: str) -> tuple[ProgramLabel, str, str]:
    normalized = _normalize_algorithm(value)
    if normalized in _NODE_WEAK_EC_CURVES:
        return (
            "misuse",
            "short_ec_curve",
            f"EC curve {value!r} has legacy or short strength",
        )
    if normalized in {
        "P224",
        "P256",
        "P384",
        "P521",
        "SECP224R1",
        "SECP256K1",
        "SECP256R1",
        "SECP384R1",
        "SECP521R1",
        "PRIME256V1",
    }:
        return "secure", "approved_ec_curve", f"EC curve {value!r} is approved"
    return (
        "unknown",
        "unclassified_ec_curve",
        f"EC curve {value!r} is not classified",
    )


def _detect_java_m04(
    analyzer: JavaAnalyzer,
    task_id: str | None,
) -> DetectionResult:
    del task_id
    if not source_may_contain_policy_sink(analyzer.source, _JAVA_SINKS):
        return evidence_result(
            analyzer,
            "M04",
            (),
            no_sink_reason="no key-size sink found",
        )
    if not analyzer.flow.ast_context.parse_success:
        return parse_failure_result(analyzer, "M04")

    evidence: list[JavaEvidence] = []
    for argument in iter_policy_arguments(analyzer.source, _JAVA_SINKS):
        if (
            argument.spec.kind == "receiver_method"
            and analyzer.int_value(argument.expression, argument.start) is None
        ):
            continue
        label, reason = _classify_java_key_size(analyzer, argument)
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
        "M04",
        evidence,
        no_sink_reason="no key-size sink found",
    )


def _classify_java_key_size(
    analyzer: JavaAnalyzer,
    argument: JavaPolicyArgument,
) -> tuple[ProgramLabel, str]:
    if argument.spec.id == "ec_gen_parameter_spec_curve":
        curve = analyzer.string_value(argument.expression, argument.start)
        if curve is None:
            return "unknown", "EC curve could not be resolved"
        if normalize_algorithm(curve) in _JAVA_WEAK_EC_CURVES:
            return "misuse", f"EC curve {curve!r} provides short or legacy strength"
        return (
            "secure",
            f"EC curve {curve!r} is not in the configured weak-curve list",
        )

    algorithm = _java_key_size_algorithm(analyzer, argument)
    size = analyzer.int_value(argument.expression, argument.start)
    if size is None:
        return (
            "unknown",
            f"{argument.spec.sink_name} key size could not be resolved",
        )
    minimum = (
        _JAVA_MINIMUM_KEY_SIZES.get(normalize_algorithm(algorithm))
        if algorithm is not None
        else None
    )
    if minimum is None:
        return (
            "unknown",
            f"minimum key size for algorithm {algorithm or 'unknown'} is not configured",
        )
    if size < minimum:
        return (
            "misuse",
            f"{algorithm} key size {size} bits is below minimum {minimum} bits",
        )
    return (
        "secure",
        f"{algorithm} key size {size} bits meets minimum {minimum} bits",
    )


def _java_key_size_algorithm(
    analyzer: JavaAnalyzer,
    argument: JavaPolicyArgument,
) -> str | None:
    if argument.spec.id == "rsa_key_gen_parameter_spec_size":
        return "RSA"
    if argument.receiver_algorithm is not None:
        return analyzer.string_value(
            argument.receiver_algorithm,
            argument.call_start,
        )
    return None



MISUSE_ID = "M04"


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
        implementation=_detect_m04,
    )


__all__ = ("MISUSE_ID", "detect")
