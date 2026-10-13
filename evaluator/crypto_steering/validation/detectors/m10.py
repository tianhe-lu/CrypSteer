"""M10 MAC-algorithm detector."""

from __future__ import annotations

import ast
import re

from crypto_steering.validation.java.detection import (
    JavaEvidence,
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
    ResolvedValue,
    combine_detector_findings,
)
from crypto_steering.validation.source.index import (
    CallExpression,
    CStyleAnalyzer,
    JavaAnalyzer,
    PythonAnalyzer,
    SourceAnalyzer,
)
from crypto_steering.validation.detectors.apis import (
    _is_typescript_crypto_function_call,
    _has_mac_terminal,
    _mac_from_csharp_callee,
)
from crypto_steering.validation.detectors.core import (
    _algorithm_finding,
    _contains_classified_algorithm,
    _resolved_is_explicit_mac_algorithm,
    _csharp_selected_mac_dispatch_findings,
    _csharp_local_helper_has_no_algorithm_selector,
    _classify_mac,
    _result,
    _dedupe_findings,
    _suppress_unresolved_internal_adapters,
    _suppress_proven_inactive_findings,
    _preserve_definite_inactive_findings,
    _call_argument,
    _first_call_argument,
    _normalize_algorithm,
    _preserve_called_python_function_findings,
    _python_canonical_detector_call_name,
)
from crypto_steering.validation.detectors.engine import execute_detector


_JAVA_SINKS = (
    JavaSinkSpec(
        id="mac_get_instance",
        kind="static_method",
        class_name="Mac",
        method_name="getInstance",
        sink_name="Mac.getInstance algorithm",
        argument_index=0,
    ),
    JavaSinkSpec(
        id="key_generator_get_instance_mac",
        kind="static_method",
        class_name="KeyGenerator",
        method_name="getInstance",
        sink_name="KeyGenerator.getInstance MAC algorithm",
        argument_index=0,
    ),
    JavaSinkSpec(
        id="secret_key_spec_mac_algorithm",
        kind="constructor",
        class_name="SecretKeySpec",
        sink_name="new SecretKeySpec MAC algorithm",
        argument_index=1,
    ),
)
_JAVA_INSECURE_MACS = {
    normalize_algorithm(value)
    for value in (
        "MD5",
        "SHA1",
        "SHA-1",
        "HmacMD5",
        "HMAC-MD5",
        "HmacSHA1",
        "HMAC-SHA1",
        "HmacSHA-1",
        "PBEWithHmacSHA1",
    )
}
_JAVA_SECURE_MACS = {
    normalize_algorithm(value)
    for value in (
        "HmacSHA224",
        "HmacSHA256",
        "HmacSHA384",
        "HmacSHA512",
        "HmacSHA3-224",
        "HmacSHA3-256",
        "HmacSHA3-384",
        "HmacSHA3-512",
        "AESCMAC",
        "KMAC128",
        "KMAC256",
    )
}


def _detect_m10(analyzer: SourceAnalyzer, task_id: str | None, modern: int) -> DetectionResult:
    if isinstance(analyzer, JavaAnalyzer):
        return _detect_java_m10(analyzer, task_id)
    del task_id, modern
    findings: list[DetectorFinding] = []
    if analyzer.language == "typescript":
        for call in analyzer.calls(
            "createHmac",
            "createIncrementalHmac",
            "createNodeHmac",
        ):
            if not _is_typescript_crypto_function_call(
                analyzer,
                call,
                {"createHmac", "createIncrementalHmac", "createNodeHmac"},
            ):
                continue
            if call.arguments:
                resolved = analyzer.resolve_strings(call.arguments[0], call.start)
                findings.append(_algorithm_finding(analyzer, "M10", call, call.arguments[0], resolved, _classify_mac))
    elif analyzer.language == "csharp":
        dispatch_findings, selected_dispatch_spans = (
            _csharp_selected_mac_dispatch_findings(analyzer)
        )
        findings.extend(dispatch_findings)
        findings.extend(
            _csharp_hmac_terminal_findings(
                analyzer,
                selected_dispatch_spans=selected_dispatch_spans,
            )
        )
        for call in analyzer.calls("CreateInstance"):
            compact = re.sub(r"\s+", "", call.callee)
            if not compact.endswith("Activator.CreateInstance") or not call.arguments:
                continue
            expression = call.arguments[0]
            resolved = analyzer.resolve_strings(expression, call.start)
            if not _resolved_is_explicit_mac_algorithm(resolved):
                continue
            findings.append(
                _algorithm_finding(
                    analyzer,
                    "M10",
                    call,
                    expression,
                    resolved,
                    _classify_mac,
                )
            )
        for call in analyzer.calls("HashData"):
            if any(start <= call.start < end for start, end in selected_dispatch_spans):
                continue
            algorithm = _mac_from_csharp_callee(call.callee)
            if algorithm:
                resolved = ResolvedValue(algorithm, "constant", (algorithm,))
                findings.append(
                    _algorithm_finding(
                        analyzer,
                        "M10",
                        call,
                        algorithm,
                        resolved,
                        _classify_mac,
                    )
                )
        for call in analyzer.calls("Create", "CreateFromName"):
            if any(start <= call.start < end for start, end in selected_dispatch_spans):
                continue
            compact = re.sub(r"\s+", "", call.callee)
            if not (
                compact.endswith("HashAlgorithm.Create")
                or compact.endswith("CryptoConfig.CreateFromName")
            ):
                continue
            expression = _call_argument(call, 0, "name", "algorithm") or ""
            if expression:
                resolved = analyzer.resolve_strings(expression, call.start)
                if not _resolved_is_explicit_mac_algorithm(resolved):
                    continue
                findings.append(
                    _algorithm_finding(
                        analyzer,
                        "M10",
                        call,
                        expression,
                        resolved,
                        _classify_mac,
                    )
                )
        for call in analyzer.calls(
            "HMACMD5", "HMACSHA1", "HMACSHA256", "HMACSHA384", "HMACSHA512",
            "HMACSHA3_256", "HMACSHA3_384", "HMACSHA3_512",
            "CreateHmac", "CreateIncrementalHmac", "CreateNativeHmac",
            "CreateHMAC",
        ):
            simple = call.simple_name
            inside_selected_dispatch = any(
                start <= call.start < end
                for start, end in selected_dispatch_spans
            )
            if inside_selected_dispatch:
                continue
            if simple.startswith("HMAC") and simple != "CreateHMAC":
                if not _csharp_native_hmac_constructor_is_terminal_bound(
                    analyzer,
                    call,
                ):
                    continue
            if (
                not simple.startswith("HMAC")
                and _csharp_local_helper_has_no_algorithm_selector(
                    analyzer,
                    simple,
                )
            ):
                # A source-defined CreateHmac(key) wrapper fixes its
                # algorithm inside the helper. The concrete constructor is
                # the policy sink; key material is not an algorithm selector.
                continue
            if simple.startswith("HMAC") and simple != "CreateHMAC":
                resolved = ResolvedValue(simple, "constant", (simple,))
                expression = simple
            elif (expression := _call_argument(
                call,
                0,
                "hashAlgorithm",
                "algorithm",
            )) is not None:
                resolved = analyzer.resolve_strings(expression, call.start)
                if simple == "CreateHMAC" and expression.startswith("new HashAlgorithmName"):
                    inner = _first_call_argument(expression)
                    if inner:
                        resolved = analyzer.resolve_strings(inner, call.start)
            else:
                continue
            if (
                not simple.startswith("HMAC")
                and not _contains_classified_algorithm(resolved, _classify_mac)
            ):
                # Local helpers named CreateHmac commonly accept only key
                # material. Their native HMAC constructor is the policy sink;
                # the key argument is not an algorithm selector.
                continue
            findings.append(_algorithm_finding(analyzer, "M10", call, expression, resolved, _classify_mac))
    elif analyzer.language == "python":
        for call in analyzer.calls("HMAC"):
            canonical = _python_canonical_detector_call_name(analyzer, call)
            if canonical not in {
                "hmac.HMAC",
                "cryptography.hazmat.primitives.hmac.HMAC",
            }:
                continue
            expression = _python_hmac_algorithm_expression(analyzer, call)
            if expression:
                resolved = _resolve_python_hmac_algorithm(
                    analyzer,
                    expression,
                    call.start,
                )
                finding = _algorithm_finding(
                    analyzer,
                    "M10",
                    call,
                    expression,
                    resolved,
                    _classify_mac,
                )
                findings.append(finding)
        for call in analyzer.calls("new", "digest", "hexdigest"):
            canonical = _python_canonical_detector_call_name(analyzer, call)
            if canonical not in {"hmac.new", "hmac.digest"}:
                continue
            expression = call.named_arguments.get("digestmod")
            if expression is None and len(call.arguments) > 2:
                expression = call.arguments[2]
            if expression:
                resolved = _resolve_python_hmac_algorithm(
                    analyzer,
                    expression,
                    call.start,
                )
                finding = _algorithm_finding(
                    analyzer,
                    "M10",
                    call,
                    expression,
                    resolved,
                    _classify_mac,
                )
                findings.append(finding)
        for call in analyzer.calls():
            expression = call.named_arguments.get("digestmod")
            if (
                expression is None
                or not _python_call_routes_to_hmac_factory(analyzer, call)
                or _python_call_is_nested_in_hmac_factory(analyzer, call)
            ):
                continue
            resolved = _resolve_python_hmac_algorithm(
                analyzer,
                expression,
                call.start,
            )
            findings.append(
                _algorithm_finding(
                    analyzer,
                    "M10",
                    call,
                    expression,
                    resolved,
                    _classify_mac,
                )
            )
    lexical_findings = tuple(findings)
    findings = _suppress_proven_inactive_findings(analyzer, findings)
    findings = _preserve_called_python_function_findings(
        analyzer,
        lexical_findings,
        findings,
    )
    findings = _preserve_definite_inactive_findings(
        analyzer,
        lexical_findings,
        findings,
    )
    if analyzer.language == "typescript":
        findings = _suppress_unresolved_internal_adapters(
            findings,
            adapter_names={"createNodeHmac"},
        )
    elif analyzer.language == "csharp":
        findings = _suppress_unresolved_internal_adapters(
            findings,
            adapter_names={"CreateNativeHmac", "CreateHMAC"},
        )
    if findings and not _has_mac_terminal(analyzer):
        findings = []
    return _result(analyzer, "M10", _dedupe_findings(findings), "no completed MAC algorithm sink found")


def _detect_java_m10(
    analyzer: JavaAnalyzer,
    task_id: str | None,
) -> DetectionResult:
    del task_id
    if not source_may_contain_policy_sink(analyzer.source, _JAVA_SINKS):
        return evidence_result(
            analyzer,
            "M10",
            (),
            no_sink_reason="no MAC algorithm sink found",
        )
    if not analyzer.flow.ast_context.parse_success:
        return parse_failure_result(analyzer, "M10")

    evidence: list[JavaEvidence] = []
    for argument in iter_policy_arguments(analyzer.source, _JAVA_SINKS):
        if argument.spec.class_name != "Mac":
            algorithm = analyzer.string_value(
                argument.expression,
                argument.start,
            )
            if algorithm is None or not _java_looks_like_mac_algorithm(algorithm):
                continue
        algorithms = analyzer.string_values(
            argument.expression,
            argument.start,
        )
        label, reason = _classify_java_mac_algorithms(
            argument.spec.sink_name,
            algorithms,
        )
        label = (
            _refine_java_m10_parameter_label(
                analyzer,
                argument.expression,
                argument.start,
            )
            or label
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
    pre_reachability_authoritative = tuple(
        item for item in evidence if "Mac.getInstance" in item.sink_name
    )
    if pre_reachability_authoritative:
        filtered_reason = (
            _java_m10_legacy_summary(pre_reachability_authoritative)
            + "; classification is based on Mac.getInstance"
        )
        result = evidence_result(
            analyzer,
            "M10",
            evidence,
            no_sink_reason=filtered_reason,
            preserve_filtered_summary=False,
        )
    else:
        result = evidence_result(
            analyzer,
            "M10",
            evidence,
            no_sink_reason="no MAC algorithm sink found",
        )
    authoritative = tuple(
        finding
        for finding in result.findings
        if "Mac.getInstance" in finding.sink_name
    )
    if not authoritative:
        return result
    classified = combine_detector_findings(
        "M10",
        "java",
        authoritative,
        no_sink_reason="no completed MAC algorithm sink found",
    )
    return DetectionResult(
        misuse_id="M10",
        language="java",
        label=classified.label,
        reason=(
            f"{classified.reason}; classification is based on Mac.getInstance"
        ),
        findings=result.findings,
    )


def _classify_java_mac_algorithms(
    sink_name: str,
    algorithms: frozenset[str] | None,
) -> tuple[ProgramLabel, str]:
    if algorithms is None:
        return "unknown", f"{sink_name} algorithm could not be resolved"
    labels = {
        (
            "misuse"
            if normalize_algorithm(algorithm) in _JAVA_INSECURE_MACS
            else "secure"
            if normalize_algorithm(algorithm) in _JAVA_SECURE_MACS
            else "unknown"
        )
        for algorithm in algorithms
    }
    rendered = ", ".join(repr(item) for item in sorted(algorithms))
    if "secure" in labels and "misuse" in labels:
        return (
            "mixed",
            f"resolved MAC algorithms include secure and broken values: {rendered}",
        )
    if "misuse" in labels:
        return (
            "misuse",
            f"at least one resolved MAC algorithm is broken or deprecated: {rendered}",
        )
    if labels == {"secure"}:
        return (
            "secure",
            f"all resolved MAC algorithms are acceptable under configured policy: {rendered}",
        )
    return (
        "unknown",
        f"at least one resolved MAC algorithm is not classified: {rendered}",
    )


def _java_m10_legacy_summary(evidence: tuple[JavaEvidence, ...]) -> str:
    labels = {item.label for item in evidence}
    if "mixed" in labels or {"secure", "misuse"} <= labels:
        return "both secure and misuse sinks were found"
    if "misuse" in labels:
        return "at least one sink uses a hardcoded or predictable source"
    if labels == {"secure"}:
        return "all sinks use non-constant sources"
    return "sinks found, but sources were unresolved"


def _java_looks_like_mac_algorithm(algorithm: str) -> bool:
    normalized = normalize_algorithm(algorithm)
    return (
        normalized.startswith("HMAC")
        or normalized.startswith("PBEWITHHMAC")
        or normalized.startswith("KMAC")
        or normalized.endswith("CMAC")
    )


def _refine_java_m10_parameter_label(
    analyzer: JavaAnalyzer,
    expression: str,
    index: int,
) -> ProgramLabel | None:
    values = analyzer.reachable_parameter_values(expression, index)
    if not values:
        return None
    labels: set[ProgramLabel] = set()
    for value in values:
        normalized = normalize_algorithm(value)
        if normalized in _JAVA_INSECURE_MACS:
            labels.add("misuse")
        elif normalized in _JAVA_SECURE_MACS:
            labels.add("secure")
        else:
            labels.add("unknown")
    if labels == {"secure", "misuse"}:
        return "mixed"
    if len(labels) == 1:
        return next(iter(labels))
    if "unknown" in labels:
        return "unknown"
    return None



MISUSE_ID = "M10"


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
        implementation=_detect_m10,
    )


__all__ = ("MISUSE_ID", "detect")


def _csharp_hmac_terminal_findings(
    analyzer: SourceAnalyzer,
    *,
    selected_dispatch_spans: tuple[tuple[int, int], ...] = (),
) -> list[DetectorFinding]:
    """Classify the C# HMAC object that reaches a completed MAC operation."""

    if not isinstance(analyzer, CStyleAnalyzer):
        return []
    findings: list[DetectorFinding] = []
    for call in analyzer.calls("ComputeHash", "GetHashAndReset"):
        if any(
            start <= call.start < end
            for start, end in selected_dispatch_spans
        ):
            # The selected-dispatch finding already represents this terminal.
            # Reclassifying every constructor text inside the same helper would
            # add dormant alternatives and turn a fixed caller choice mixed.
            continue
        compact = re.sub(r"\s+", "", call.callee)
        if "." not in compact:
            continue
        receiver = compact.rsplit(".", 1)[0]
        if _csharp_hmac_receiver_uses_selected_dispatch(
            analyzer,
            receiver,
            call.start,
            selected_dispatch_spans,
        ):
            continue
        algorithms = _resolve_csharp_hmac_receiver_algorithms(
            analyzer,
            receiver,
            call.start,
            seen=frozenset(),
        )
        if not algorithms:
            continue
        resolved = ResolvedValue(
            receiver,
            "constant",
            tuple(sorted(algorithms)),
        )
        finding_call = call
        if len(algorithms) == 1:
            algorithm = next(iter(algorithms))
            finding_call = CallExpression(
                callee=f"new{algorithm}",
                arguments=(),
                named_arguments={},
                start=call.start,
                end=call.end,
            )
        findings.append(
            _algorithm_finding(
                analyzer,
                "M10",
                finding_call,
                receiver,
                resolved,
                _classify_mac,
            )
        )
    return findings


def _csharp_hmac_receiver_uses_selected_dispatch(
    analyzer: CStyleAnalyzer,
    receiver: str,
    index: int,
    selected_dispatch_spans: tuple[tuple[int, int], ...],
    *,
    seen: frozenset[str] = frozenset(),
) -> bool:
    if not selected_dispatch_spans:
        return False
    reference = receiver.rsplit(".", 1)[-1]
    identity = f"{reference}:{index}"
    if identity in seen:
        return False
    next_seen = seen | frozenset({identity})

    def expression_uses_dispatch(expression: str, expression_index: int) -> bool:
        helper_call = re.fullmatch(
            r"(?:[A-Za-z_]\w*\s*\.\s*)*(?P<name>[A-Za-z_]\w*)"
            r"\s*\(.*\)",
            expression.strip(),
            flags=re.DOTALL,
        )
        if helper_call is not None and any(
            start <= definition.start < end
            for definition in analyzer.function(helper_call.group("name"))
            for start, end in selected_dispatch_spans
        ):
            return True
        stripped = expression.strip()
        argument_reference = (
            stripped
            if re.fullmatch(
                r"(?:this\s*\.\s*)?[A-Za-z_]\w*"
                r"(?:\s*\.\s*[A-Za-z_]\w*)?",
                stripped,
            )
            else None
        )
        return bool(
            argument_reference is not None
            and _csharp_hmac_receiver_uses_selected_dispatch(
                analyzer,
                argument_reference,
                expression_index,
                selected_dispatch_spans,
                seen=next_seen,
            )
        )

    assignment = analyzer.latest_assignment(reference, index)
    if assignment is not None and expression_uses_dispatch(
        assignment.expression,
        assignment.start,
    ):
        return True

    containing = next(
        (
            definition
            for definition in analyzer.functions
            if definition.start <= index < definition.end
            and reference in definition.parameters
        ),
        None,
    )
    if containing is None:
        return False
    parameter_index = containing.parameters.index(reference)
    for caller in analyzer.calls(containing.name):
        if containing.start <= caller.start < containing.end:
            continue
        if not analyzer._call_can_target_function(caller, containing):
            continue
        argument = analyzer._c_style_argument_for_parameter(
            caller,
            parameter_index,
            reference,
        )
        if argument is not None and expression_uses_dispatch(argument, caller.start):
            return True
    return False


def _resolve_csharp_hmac_receiver_algorithms(
    analyzer: CStyleAnalyzer,
    receiver: str,
    index: int,
    *,
    seen: frozenset[str],
) -> set[str]:
    reference = receiver.rsplit(".", 1)[-1]
    key = f"{reference}:{index}"
    if key in seen:
        return set()
    next_seen = seen | frozenset({key})

    direct = _mac_from_csharp_callee(receiver)
    if direct:
        return {direct}

    assignment = analyzer.latest_assignment(reference, index)
    reaching = analyzer.reaching_assignment_alternatives(reference, index)
    if reaching:
        resolved: set[str] = set()
        for candidate in reaching:
            resolved.update(
                _csharp_hmac_algorithms_from_expression(
                    analyzer,
                    candidate.expression,
                    candidate.start,
                    seen=next_seen,
                )
            )
        if resolved:
            return resolved
    elif assignment is not None:
        resolved = _csharp_hmac_algorithms_from_expression(
            analyzer,
            assignment.expression,
            assignment.start,
            seen=next_seen,
        )
        if resolved:
            return resolved

    hash_name = analyzer.latest_assignment(f"{receiver}.HashName", index)
    if hash_name is not None:
        selected = analyzer.resolve_strings(
            hash_name.expression,
            hash_name.start,
        )
        algorithms = {
            value
            for value in selected.values
            if _classify_mac(value)[0] in {"secure", "misuse"}
        }
        if algorithms:
            return algorithms

    declaration_pattern = re.compile(
        rf"\b(?P<type>HMAC(?:MD5|SHA1|SHA224|SHA256|SHA384|SHA512|"
        rf"SHA3_256|SHA3_384|SHA3_512))\s+{re.escape(reference)}\b"
        rf"(?:\s*=\s*(?P<expression>[^;]+))?\s*;"
    )
    declarations = tuple(declaration_pattern.finditer(analyzer.source[:index]))
    if declarations:
        declaration = declarations[-1]
        expression = declaration.group("expression")
        if expression:
            resolved = _csharp_hmac_algorithms_from_expression(
                analyzer,
                expression,
                declaration.start(),
                seen=next_seen,
            )
            if resolved:
                return resolved
        return {declaration.group("type")}

    member_algorithms = _csharp_member_hmac_algorithms(
        analyzer,
        reference,
        index,
        seen=next_seen,
    )
    if member_algorithms:
        return member_algorithms

    containing = next(
        (
            definition
            for definition in analyzer.functions
            if definition.start <= index < definition.end
            and reference in definition.parameters
        ),
        None,
    )
    if containing is not None:
        parameter_index = containing.parameters.index(reference)
        values: set[str] = set()
        for call in analyzer.calls(containing.name):
            if containing.start <= call.start < containing.end:
                continue
            argument = _call_argument(
                call,
                parameter_index,
                containing.parameters[parameter_index],
            )
            if argument is None or not analyzer.position_is_reachable(call.start):
                continue
            values.update(
                _csharp_hmac_algorithms_from_expression(
                    analyzer,
                    argument,
                    call.start,
                    seen=next_seen,
                )
            )
        if values:
            return values
    return set()


def _csharp_hmac_algorithms_from_expression(
    analyzer: CStyleAnalyzer,
    expression: str,
    index: int,
    *,
    seen: frozenset[str],
) -> set[str]:
    expression = re.sub(
        r"^\s*\([^()]+\)\s*",
        "",
        expression,
        count=1,
    )
    algorithms = {
        match.group(0)
        for match in re.finditer(
            r"\bHMAC(?:MD5|SHA1|SHA224|SHA256|SHA384|SHA512|"
            r"SHA3_256|SHA3_384|SHA3_512)\b",
            expression,
        )
    }
    conditional_choices = tuple(
        re.finditer(
            r"(?P<selector>[A-Za-z_]\w*)\s*==\s*\"(?P<value>[^\"]+)\""
            r"\s*\)?\s*\?\s*(?:new\s+)?(?P<algorithm>HMAC[A-Za-z0-9_]+)",
            expression,
        )
    )
    if conditional_choices:
        selector = conditional_choices[0].group("selector")
        selected = analyzer.resolve_strings(selector, index)
        if len(selected.values) == 1:
            normalized = _normalize_algorithm(selected.values[0])
            chosen = {
                choice.group("algorithm")
                for choice in conditional_choices
                if _normalize_algorithm(choice.group("value")) == normalized
            }
            if chosen:
                return chosen
            fallback = re.search(
                r":\s*(?:new\s+)?(?P<algorithm>HMAC[A-Za-z0-9_]+)\s*\([^?:;]*\)\s*$",
                expression,
            )
            if fallback is not None:
                return {fallback.group("algorithm")}

    factory = re.fullmatch(
        r"(?:System\s*\.\s*Security\s*\.\s*Cryptography\s*\.)?"
        r"HMAC\s*\.\s*Create\s*\((?P<arguments>.*)\)",
        expression.strip(),
        flags=re.DOTALL,
    )
    if factory is not None:
        selector = _first_call_argument(expression)
        if selector:
            resolved = analyzer.resolve_strings(selector, index)
            classified = {
                value
                for value in resolved.values
                if _classify_mac(value)[0] in {"secure", "misuse"}
            }
            if classified:
                return classified

    helper_call = re.fullmatch(
        r"(?:[A-Za-z_]\w*\s*\.\s*)*(?P<name>[A-Za-z_]\w*)\s*"
        r"\((?P<arguments>.*)\)",
        expression.strip(),
        flags=re.DOTALL,
    )
    if helper_call is not None:
        helper_name = helper_call.group("name")
        helper_key = f"helper:{helper_name}:{index}"
        if helper_key not in seen:
            returned_algorithms: set[str] = set()
            for definition in analyzer.function(helper_name):
                for returned in definition.returns:
                    returned_algorithms.update(
                        _csharp_hmac_algorithms_from_expression(
                            analyzer,
                            returned.expression,
                            returned.index,
                            seen=seen | frozenset({helper_key}),
                        )
                    )
            if returned_algorithms:
                return returned_algorithms

    reference_match = re.fullmatch(
        r"[A-Za-z_]\w*(?:\s*\.\s*[A-Za-z_]\w*)*",
        expression.strip(),
    )
    if reference_match is not None:
        reference = re.sub(r"\s+", "", reference_match.group(0))
        return _resolve_csharp_hmac_receiver_algorithms(
            analyzer,
            reference,
            index,
            seen=seen,
        )
    return algorithms


def _csharp_member_hmac_algorithms(
    analyzer: CStyleAnalyzer,
    reference: str,
    index: int,
    *,
    seen: frozenset[str],
) -> set[str]:
    """Resolve an HMAC field initialized by its containing class constructor.

    A terminal such as ``_hmac.ComputeHash(...)`` commonly lives in a method
    separate from ``_hmac = new HMACSHA256(...)``. Local-assignment lookup is
    intentionally scope-safe, so detector-level object-state resolution must
    join those two sites within the same class.
    """

    owner = analyzer._containing_class_name(index)
    if owner is None:
        return set()
    compact_reference = re.sub(r"\s+", "", reference)
    compact_reference = re.sub(r"^(?:this|base)\.", "", compact_reference)
    values: set[str] = set()
    for assignment in analyzer.assignments:
        candidate = re.sub(r"\s+", "", assignment.name)
        candidate = re.sub(r"^(?:this|base)\.", "", candidate)
        if candidate != compact_reference:
            continue
        if analyzer._containing_class_name(assignment.start) != owner:
            continue
        scope = analyzer._assignment_scopes[assignment]
        if scope is not None and scope.name not in {owner, "constructor"}:
            continue
        values.update(
            _csharp_hmac_algorithms_from_expression(
                analyzer,
                assignment.expression,
                assignment.start,
                seen=seen,
            )
        )
    return values


def _csharp_native_hmac_constructor_is_terminal_bound(
    analyzer: SourceAnalyzer,
    constructor: CallExpression,
) -> bool:
    """Keep only an inline native constructor; assigned objects use terminals."""

    statement_end = analyzer.source.find(";", constructor.end)
    if statement_end == -1:
        statement_end = len(analyzer.source)
    suffix = analyzer.source[constructor.end:statement_end]
    return bool(re.search(r"\.\s*(?:ComputeHash|GetHashAndReset)\s*\(", suffix))


def _python_hmac_algorithm_expression(
    analyzer: SourceAnalyzer,
    call: CallExpression,
) -> str | None:
    aliases = getattr(analyzer, "import_aliases", {})
    root, separator, suffix = call.callee.partition(".")
    canonical = aliases.get(root, root) + (separator + suffix if separator else "")
    if canonical in {"hmac.HMAC", "hmac.new"}:
        return call.named_arguments.get("digestmod") or (
            call.arguments[2] if len(call.arguments) > 2 else None
        )
    return call.named_arguments.get("algorithm") or (
        call.arguments[1] if len(call.arguments) > 1 else None
    )


def _python_call_routes_to_hmac_factory(
    analyzer: SourceAnalyzer,
    call: CallExpression,
) -> bool:
    if not isinstance(analyzer, PythonAnalyzer):
        return False
    target = analyzer.resolve_call_target_strings(call)
    expressions = (
        call.callee,
        target.expression,
        *(step.expression for step in target.steps),
    )
    return any(
        re.fullmatch(r"(?:hmac\s*\.\s*)?(?:new|digest)", value.strip())
        is not None
        for value in expressions
    )


def _python_call_is_nested_in_hmac_factory(
    analyzer: PythonAnalyzer,
    call: CallExpression,
) -> bool:
    """Exclude a factory-like argument nested inside another HMAC call."""

    for outer in analyzer.calls():
        if not (outer.start < call.start and call.end <= outer.end):
            continue
        if _python_canonical_detector_call_name(analyzer, outer) in {
            "hmac.HMAC",
            "hmac.new",
            "hmac.digest",
        }:
            return True
        if _python_call_routes_to_hmac_factory(analyzer, outer):
            return True
    return False


def _resolve_python_hmac_algorithm(
    analyzer: SourceAnalyzer,
    expression: str,
    index: int,
) -> ResolvedValue:
    """Resolve digest identity through narrow generated-code wrappers."""

    resolved = analyzer.resolve_strings(expression, index)
    if _contains_classified_algorithm(resolved, _classify_mac):
        return resolved
    if not isinstance(analyzer, PythonAnalyzer):
        return resolved

    assignment = _python_latest_reference_assignment_expression(
        analyzer.source,
        expression,
        index,
    )
    candidate = assignment[0] if assignment is not None else expression
    candidate_index = assignment[1] if assignment is not None else index
    try:
        node = ast.parse(candidate, mode="eval").body
    except SyntaxError:
        return resolved
    selector = _python_hmac_selector_expression(node)
    if selector is None:
        return resolved
    selected = analyzer.resolve_strings(selector, candidate_index)
    return selected if selected.values else resolved


def _python_hmac_selector_expression(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    name = ast.unparse(node.func)
    if (
        name in {"hashlib.getdigestmodule", "hashlib.algorithms.get"}
        and node.args
    ):
        return ast.unparse(node.args[0])
    if (
        isinstance(node.func, ast.Attribute)
        and node.func.attr in {"digest", "hexdigest"}
        and isinstance(node.func.value, ast.Call)
    ):
        return _python_hmac_selector_expression(node.func.value)
    if (
        isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) > 1
    ):
        return ast.unparse(node.args[1])
    if isinstance(node.func, ast.Call):
        selector_call = node.func
        if (
            isinstance(selector_call.func, ast.Name)
            and selector_call.func.id == "getattr"
            and len(selector_call.args) > 1
        ):
            return ast.unparse(selector_call.args[1])
    return None


def _python_latest_reference_assignment_expression(
    source: str,
    reference: str,
    index: int,
) -> tuple[str, int] | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    line_offsets = [0]
    line_offsets.extend(match.end() for match in re.finditer(r"\n", source))
    candidates: list[tuple[int, ast.AST]] = []
    for statement in ast.walk(tree):
        if isinstance(statement, ast.Assign):
            value = statement.value
            targets = tuple(statement.targets)
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            value = statement.value
            targets = (statement.target,)
        else:
            continue
        if not any(ast.unparse(target) == reference for target in targets):
            continue
        start = line_offsets[statement.lineno - 1] + statement.col_offset
        if start < index:
            candidates.append((start, value))
    if not candidates:
        return None
    start, value = max(candidates, key=lambda item: item[0])
    return ast.get_source_segment(source, value) or ast.unparse(value), start


def _python_hmac_digestmod_is_hash_instance(
    analyzer: SourceAnalyzer,
    expression: str,
    index: int,
    *,
    seen: frozenset[str] = frozenset(),
) -> bool:
    """Identify the definite ``hashlib.new(...)`` object form.

    ``hmac.new`` accepts a digest name or constructor. A constructed hash
    object occupies the same argument position but cannot be used to complete
    the HMAC operation, so its security algorithm is not treated as resolved.
    """

    compact = expression.strip()
    state_key = f"{index}:{compact}"
    if state_key in seen:
        return False
    next_seen = seen | frozenset({state_key})
    try:
        node = ast.parse(compact, mode="eval").body
    except SyntaxError:
        return False
    if isinstance(node, ast.Call):
        call_name = ast.unparse(node.func)
        # ``getattr(hashlib, name)`` and mapping ``.get(...)`` calls return a
        # digest constructor. Calling that returned constructor would create
        # the hash instance, but merely selecting it is a valid digestmod.
        return call_name != "getattr" and not call_name.endswith(".get")
    if not isinstance(node, ast.Name):
        return False
    assignment = _python_latest_name_assignment_expression(
        analyzer.source,
        node.id,
        index,
    )
    if assignment is None:
        return False
    assignment_expression, assignment_start = assignment
    return _python_hmac_digestmod_is_hash_instance(
        analyzer,
        assignment_expression,
        assignment_start,
        seen=next_seen,
    )


def _python_latest_name_assignment_expression(
    source: str,
    name: str,
    index: int,
) -> tuple[str, int] | None:
    """Return the latest simple Python assignment before a sink."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    line_offsets = [0]
    line_offsets.extend(match.end() for match in re.finditer(r"\n", source))
    candidates: list[tuple[int, ast.AST]] = []
    for statement in ast.walk(tree):
        value: ast.AST | None = None
        targets: tuple[ast.AST, ...] = ()
        if isinstance(statement, ast.Assign):
            value = statement.value
            targets = tuple(statement.targets)
        elif isinstance(statement, ast.AnnAssign):
            value = statement.value
            targets = (statement.target,)
        if value is None or not any(
            isinstance(target, ast.Name) and target.id == name
            for target in targets
        ):
            continue
        start = line_offsets[statement.lineno - 1] + statement.col_offset
        if start < index:
            candidates.append((start, value))
    if not candidates:
        return None
    start, value = max(candidates, key=lambda item: item[0])
    return ast.get_source_segment(source, value) or ast.unparse(value), start


__all__ = ("MISUSE_ID", "detect")
