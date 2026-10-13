"""M09 hash-algorithm detector."""

from __future__ import annotations

import re

from crypto_steering.validation.java.calls import static_method_calls
from crypto_steering.validation.java.detection import (
    JavaEvidence,
    JavaPolicyArgument,
    JavaSinkSpec,
    evidence_result,
    iter_policy_arguments,
    source_may_contain_policy_sink,
)
from crypto_steering.validation.java.lightweight_ast import (
    method_containing_index,
    parse_method_nodes,
)
from crypto_steering.validation.java.resolution import (
    java_string_literal_value,
    normalize_algorithm,
)
from crypto_steering.validation.java.source import (
    find_matching_delimiter,
    mask_java_comments_and_literals,
    strip_java_comments,
)
from crypto_steering.validation.java.static_analysis import (
    simple_variable,
    strip_outer_expression,
)
from crypto_steering.validation.models import (
    DetectionResult,
    DetectorFinding,
    Language,
    ProgramLabel,
    ResolvedValue,
)
from crypto_steering.validation.source.index import (
    Assignment,
    CStyleAnalyzer,
    CallExpression,
    JavaAnalyzer,
    PythonAnalyzer,
    SourceAnalyzer,
)
from crypto_steering.validation.detectors.apis import (
    _is_typescript_crypto_function_call,
    _has_hash_terminal,
    _algorithm_from_csharp_callee,
    _typescript_signature_digest_findings,
    _csharp_signature_digest_findings,
    _python_signature_digest_findings,
)
from crypto_steering.validation.detectors.core import (
    _algorithm_finding,
    _contains_classified_algorithm,
    _csharp_selected_hash_dispatch_findings,
    _typescript_guarded_algorithm_values,
    _classify_digest,
    _result,
    _dedupe_findings,
    _suppress_proven_inactive_findings,
    _preserve_definite_inactive_findings,
    _call_argument,
    _normalize_algorithm,
    _preserve_called_python_function_findings,
)
from crypto_steering.validation.detectors.engine import execute_detector


_JAVA_SINKS = (
    JavaSinkSpec(
        id="message_digest_get_instance",
        kind="static_method",
        class_name="MessageDigest",
        method_name="getInstance",
        sink_name="MessageDigest.getInstance algorithm",
        argument_index=0,
    ),
    JavaSinkSpec(
        id="signature_get_instance",
        kind="static_method",
        class_name="Signature",
        method_name="getInstance",
        sink_name="Signature.getInstance digest algorithm",
        argument_index=0,
    ),
)
_JAVA_INSECURE_DIGESTS = {
    normalize_algorithm(value)
    for value in ("MD2", "MD4", "MD5", "SHA", "SHA-1", "SHA1")
}
_JAVA_SECURE_DIGESTS = {
    normalize_algorithm(value)
    for value in (
        "SHA-256",
        "SHA256",
        "SHA-384",
        "SHA384",
        "SHA-512",
        "SHA512",
        "SHA3-224",
        "SHA3-256",
        "SHA3-384",
        "SHA3-512",
        "BLAKE2b-256",
        "BLAKE2b-512",
        "BLAKE2s-256",
    )
}
_JAVA_INSECURE_SIGNATURE_FRAGMENTS = {
    normalize_algorithm(value)
    for value in ("MD2with", "MD5with", "SHA1with", "SHA-1with")
}
_JAVA_SECURE_SIGNATURE_FRAGMENTS = {
    normalize_algorithm(value)
    for value in (
        "SHA256with",
        "SHA384with",
        "SHA512with",
        "SHA3-256with",
        "SHA3-384with",
        "SHA3-512with",
    )
}


def _detect_m09(analyzer: SourceAnalyzer, task_id: str | None, modern: int) -> DetectionResult:
    if isinstance(analyzer, JavaAnalyzer):
        return _detect_java_m09(analyzer, task_id)
    del modern
    findings: list[DetectorFinding] = []
    if analyzer.language == "typescript":
        for call in analyzer.calls("createHash"):
            if not _is_typescript_crypto_function_call(
                analyzer,
                call,
                {"createHash"},
            ):
                continue
            if call.arguments:
                if not _typescript_hash_context_reaches_digest(analyzer, call):
                    continue
                resolved = analyzer.resolve_strings(call.arguments[0], call.start)
                if not _contains_classified_algorithm(resolved, _classify_digest):
                    guarded_values = _typescript_guarded_algorithm_values(
                        analyzer,
                        call,
                        call.arguments[0],
                        _classify_digest,
                    )
                    if guarded_values:
                        resolved = ResolvedValue(
                            call.arguments[0],
                            "runtime_input",
                            guarded_values,
                            unresolved_reason=(
                                "runtime selection is restricted by a terminating "
                                "TypeScript string-union guard"
                            ),
                        )
                findings.append(_algorithm_finding(analyzer, "M09", call, call.arguments[0], resolved, _classify_digest))
        findings.extend(_typescript_signature_digest_findings(analyzer))
    elif analyzer.language == "csharp":
        dispatch_findings, selected_dispatch_spans = (
            _csharp_selected_hash_dispatch_findings(analyzer)
        )
        findings.extend(dispatch_findings)
        inline_findings, inline_dispatch_spans = (
            _csharp_selected_inline_hash_findings(analyzer)
        )
        findings.extend(inline_findings)
        runtime_inline_findings, runtime_inline_spans = (
            _csharp_runtime_inline_hash_dispatch_findings(analyzer)
        )
        findings.extend(runtime_inline_findings)
        selected_dispatch_spans = tuple(
            sorted(
                {
                    *selected_dispatch_spans,
                    *inline_dispatch_spans,
                    *runtime_inline_spans,
                }
            )
        )
        for call in analyzer.calls("HashData", "ComputeHash"):
            if any(
                start <= call.start < end
                for start, end in selected_dispatch_spans
            ):
                continue
            if "HMAC" in call.callee:
                continue
            if "CryptographicOperations.HashData" in call.callee and call.arguments:
                expression = _call_argument(
                    call,
                    0,
                    "hashAlgorithm",
                    "algorithm",
                ) or ""
                resolved = analyzer.resolve_strings(expression, call.start)
            else:
                expression = _algorithm_from_csharp_callee(call.callee)
                resolved = ResolvedValue(
                    expression,
                    "constant",
                    (expression,),
                )
            if expression:
                findings.append(_algorithm_finding(analyzer, "M09", call, expression, resolved, _classify_digest))
        for call in analyzer.calls("CreateHash"):
            if (
                "IncrementalHash.CreateHash" not in re.sub(r"\s+", "", call.callee)
                or not call.arguments
            ):
                continue
            expression = _call_argument(
                call,
                0,
                "hashAlgorithm",
                "algorithm",
            ) or ""
            if expression:
                resolved = analyzer.resolve_strings(expression, call.start)
                findings.append(
                    _algorithm_finding(
                        analyzer,
                        "M09",
                        call,
                        expression,
                        resolved,
                        _classify_digest,
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
                findings.append(
                    _algorithm_finding(
                        analyzer,
                        "M09",
                        call,
                        expression,
                        resolved,
                        _classify_digest,
                    )
                )
        for call in analyzer.calls("Create"):
            if any(start <= call.start < end for start, end in selected_dispatch_spans):
                continue
            algorithm = _algorithm_from_csharp_callee(call.callee)
            if algorithm in {
                "MD5", "SHA1", "SHA224", "SHA256", "SHA384", "SHA512",
                "SHA3_256", "SHA3_384", "SHA3_512",
            }:
                resolved = ResolvedValue(algorithm, "constant", (algorithm,))
                findings.append(_algorithm_finding(analyzer, "M09", call, algorithm, resolved, _classify_digest))
        for call in analyzer.calls(
            "MD5CryptoServiceProvider",
            "SHA1CryptoServiceProvider",
            "SHA1Managed",
            "SHA256CryptoServiceProvider",
            "SHA256Managed",
            "SHA384Managed",
            "SHA512Managed",
        ):
            algorithm = _algorithm_from_csharp_callee(call.callee)
            if algorithm:
                resolved = ResolvedValue(algorithm, "constant", (algorithm,))
                findings.append(
                    _algorithm_finding(
                        analyzer,
                        "M09",
                        call,
                        algorithm,
                        resolved,
                        _classify_digest,
                    )
                )
        findings.extend(_csharp_reflection_hash_findings(analyzer))
        findings.extend(_csharp_signature_digest_findings(analyzer))
    elif analyzer.language == "python":
        handled_call_starts: set[int] = set()
        hmac_ranges = tuple(
            (call.start, call.end)
            for call in analyzer.calls("HMAC", "new", "digest")
            if "hmac" in call.callee.lower()
        )
        for call in analyzer.calls("Hash"):
            if call.arguments:
                resolved = analyzer.resolve_strings(call.arguments[0], call.start)
                findings.append(_algorithm_finding(analyzer, "M09", call, call.arguments[0], resolved, _classify_digest))
                handled_call_starts.add(call.start)
        for call in analyzer.calls(
            "md5", "sha1", "sha224", "sha256", "sha384", "sha512",
            "sha3_224", "sha3_256", "sha3_384", "sha3_512",
            "blake2b", "blake2s", "new",
        ):
            if any(start < call.start < end for start, end in hmac_ranges):
                continue
            if "hmac" in call.callee.lower():
                continue
            if call.simple_name == "new" and call.callee != "hashlib.new":
                continue
            if (
                call.simple_name == "new"
                and isinstance(analyzer, PythonAnalyzer)
                and analyzer.call_is_container_literal_value(call)
            ):
                continue
            expression = (
                call.callee
                if call.simple_name != "new"
                else (call.arguments[0] if call.arguments else "")
            )
            if expression:
                resolved = (
                    analyzer.resolve_call_target_strings(call)
                    if call.simple_name != "new"
                    else analyzer.resolve_strings(expression, call.start)
                )
                findings.append(_algorithm_finding(analyzer, "M09", call, expression, resolved, _classify_digest))
                handled_call_starts.add(call.start)
        for call in analyzer.calls():
            is_hashlib_callable = _is_python_hashlib_callable(analyzer, call)
            if (
                call.start in handled_call_starts
                or "hmac" in call.callee.lower()
                or any(start < call.start < end for start, end in hmac_ranges)
                or (
                    analyzer.analysis_profile == "reachable_security_flow"
                    and isinstance(analyzer, PythonAnalyzer)
                    and _python_m09_algorithm_descriptor_call(analyzer, call)
                )
                or (
                    call.simple_name
                    in {
                        "digest",
                        "hexdigest",
                        "update",
                        "finalize",
                        "verify",
                        "copy",
                        "decode",
                        "encode",
                    }
                    and (
                        not is_hashlib_callable
                        or _python_call_is_terminal_method(analyzer, call)
                    )
                )
            ):
                continue
            resolved = analyzer.resolve_call_target_strings(call)
            if (
                not _contains_classified_algorithm(resolved, _classify_digest)
                and not is_hashlib_callable
            ):
                continue
            expression = call.callee or analyzer.source[call.start : call.end]
            findings.append(
                _algorithm_finding(
                    analyzer,
                    "M09",
                    call,
                    expression,
                    resolved,
                    _classify_digest,
                )
            )
        findings.extend(_python_signature_digest_findings(analyzer))
    if task_id and not task_id.startswith(("F4", "F5")):
        findings = [finding for finding in findings if finding.metadata.get("signature_context")]
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
    if findings and not _has_hash_terminal(analyzer):
        findings = []
    return _result(analyzer, "M09", _dedupe_findings(findings), "no completed cryptographic hash sink found")


def _detect_java_m09(
    analyzer: JavaAnalyzer,
    task_id: str | None,
) -> DetectionResult:
    del task_id
    if not source_may_contain_policy_sink(analyzer.source, _JAVA_SINKS):
        return evidence_result(
            analyzer,
            "M09",
            (),
            no_sink_reason="no hash algorithm sink found",
        )

    if analyzer.flow.ast_context.parse_success:
        evidence = _java_m09_parsed_evidence(analyzer)
    else:
        evidence = _java_m09_modern_fallback_evidence(analyzer.source)
    evidence = [
        JavaEvidence(
            sink_name=item.sink_name,
            expression=item.expression,
            label=(
                _refine_java_m09_parameter_label(
                    analyzer,
                    item.sink_name,
                    item.expression,
                    item.start,
                )
                or item.label
            ),
            reason=item.reason,
            start=item.start,
            end=item.end,
        )
        for item in evidence
    ]
    return evidence_result(
        analyzer,
        "M09",
        evidence,
        no_sink_reason="no hash algorithm sink found",
    )


def _java_m09_parsed_evidence(analyzer: JavaAnalyzer) -> list[JavaEvidence]:
    evidence: list[JavaEvidence] = []
    for argument in iter_policy_arguments(analyzer.source, _JAVA_SINKS):
        algorithms = analyzer.string_values(
            argument.expression,
            argument.start,
        )
        label, reason = _classify_java_m09_argument(
            analyzer,
            argument,
            algorithms,
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
    return [
        item
        for item in evidence
        if not _java_m09_is_unreachable_algorithm_fallback(
            analyzer.source,
            item,
            evidence,
        )
    ]


def _classify_java_m09_argument(
    analyzer: JavaAnalyzer,
    argument: JavaPolicyArgument,
    algorithms: frozenset[str] | None,
) -> tuple[ProgramLabel, str]:
    if not algorithms or _java_m09_has_unresolved_runtime_assignment(
        analyzer,
        argument,
    ):
        return (
            "unknown",
            f"{argument.spec.sink_name} algorithm could not be resolved",
        )
    classifications = [
        (
            _classify_java_signature_algorithm(algorithm)
            if argument.spec.class_name == "Signature"
            else _classify_java_digest_algorithm(algorithm)
        )
        for algorithm in sorted(algorithms)
    ]
    labels = {label for label, _ in classifications}
    if len(labels) == 1:
        return (
            next(iter(labels)),
            "; ".join(reason for _, reason in classifications),
        )
    choices = ", ".join(sorted(algorithms))
    if {"secure", "misuse"} <= labels:
        return (
            "mixed",
            f"{argument.spec.sink_name} can select secure and broken "
            f"algorithms: {choices}",
        )
    if "misuse" in labels:
        return (
            "misuse",
            f"{argument.spec.sink_name} can select a broken algorithm: "
            f"{choices}",
        )
    if "unknown" in labels:
        return (
            "unknown",
            f"{argument.spec.sink_name} includes an unclassified algorithm: "
            f"{choices}",
        )
    return (
        "secure",
        f"{argument.spec.sink_name} selects only acceptable algorithms: "
        f"{choices}",
    )


def _classify_java_digest_algorithm(
    algorithm: str,
) -> tuple[ProgramLabel, str]:
    normalized = normalize_algorithm(algorithm)
    if normalized in _JAVA_INSECURE_DIGESTS:
        return "misuse", f"hash algorithm {algorithm!r} is broken or deprecated"
    if normalized in _JAVA_SECURE_DIGESTS:
        return (
            "secure",
            f"hash algorithm {algorithm!r} is acceptable under configured policy",
        )
    return "unknown", f"hash algorithm {algorithm!r} is not classified"


def _classify_java_signature_algorithm(
    algorithm: str,
) -> tuple[ProgramLabel, str]:
    normalized = normalize_algorithm(algorithm)
    if any(
        fragment in normalized
        for fragment in _JAVA_INSECURE_SIGNATURE_FRAGMENTS
    ):
        return (
            "misuse",
            f"signature algorithm {algorithm!r} uses a broken digest",
        )
    if any(
        fragment in normalized
        for fragment in _JAVA_SECURE_SIGNATURE_FRAGMENTS
    ):
        return (
            "secure",
            f"signature algorithm {algorithm!r} uses an acceptable digest",
        )
    return "unknown", f"signature algorithm {algorithm!r} is not classified"


def _refine_java_m09_parameter_label(
    analyzer: JavaAnalyzer,
    sink_name: str,
    expression: str,
    index: int,
) -> ProgramLabel | None:
    values = analyzer.reachable_parameter_values(expression, index)
    if not values:
        return None
    labels: set[ProgramLabel] = set()
    for value in values:
        if "Signature" in sink_name:
            label, _ = _classify_java_signature_algorithm(value)
        else:
            label, _ = _classify_java_digest_algorithm(value)
        labels.add(label)
    if labels == {"secure", "misuse"}:
        return "mixed"
    if len(labels) == 1:
        return next(iter(labels))
    if "unknown" in labels:
        return "unknown"
    return None


def _java_m09_is_unreachable_algorithm_fallback(
    source: str,
    item: JavaEvidence,
    evidence: list[JavaEvidence],
) -> bool:
    """Ignore a catch fallback when a fixed standard digest cannot miss."""

    masked = mask_java_comments_and_literals(source)
    for catch in reversed(tuple(re.finditer(r"\bcatch\s*\(", masked[: item.start]))):
        opening = masked.find("(", catch.start())
        closing = find_matching_delimiter(masked, opening)
        if closing == -1 or "NoSuchAlgorithmException" not in source[opening:closing]:
            continue
        body_start = masked.find("{", closing)
        if body_start == -1:
            continue
        body_end = find_matching_delimiter(masked, body_start, "{", "}")
        if not body_start < item.start < body_end:
            continue
        try_matches = tuple(re.finditer(r"\btry\s*\{", masked[: catch.start()]))
        if not try_matches:
            return False
        try_start = try_matches[-1].end() - 1
        try_end = find_matching_delimiter(masked, try_start, "{", "}")
        if try_end == -1 or try_end > catch.start():
            return False
        protected = [
            candidate
            for candidate in evidence
            if try_start < candidate.start < try_end
        ]
        return bool(protected) and all(
            candidate.label in {"secure", "misuse"}
            for candidate in protected
        )
    return False


def _java_m09_has_unresolved_runtime_assignment(
    analyzer: JavaAnalyzer,
    argument: JavaPolicyArgument,
) -> bool:
    """Do not let a later literal fallback hide an earlier runtime value."""

    variable = simple_variable(strip_outer_expression(argument.expression))
    if variable is None:
        return False
    sink_method = method_containing_index(
        analyzer.flow.methods,
        argument.start,
    )
    assignments = []
    for assignment in analyzer.flow.assignments:
        if assignment.name != variable or assignment.start >= argument.start:
            continue
        assignment_method = method_containing_index(
            analyzer.flow.methods,
            assignment.start,
        )
        if sink_method is None:
            same_scope = assignment_method is None
        else:
            same_scope = bool(
                assignment_method is not None
                and assignment_method.start == sink_method.start
            )
        if same_scope:
            assignments.append(assignment)
    if len(assignments) < 2:
        return False
    runtime_pattern = re.compile(
        r"\bargs\b|\bSystem\s*\.\s*(?:getenv|getProperty)\s*\("
        r"|\bConsole\s*\.\s*readLine\s*\(|\bScanner\b",
        flags=re.IGNORECASE,
    )
    return any(
        runtime_pattern.search(assignment.expression)
        and analyzer.string_values(
            assignment.expression,
            assignment.start,
        )
        is None
        for assignment in assignments
    )


def _java_m09_modern_fallback_evidence(source: str) -> list[JavaEvidence]:
    """Resolve finite algorithm choices around unsupported Java 14+ syntax."""

    evidence: list[JavaEvidence] = []
    methods = parse_method_nodes(source)
    for spec in _JAVA_SINKS:
        for call in static_method_calls(source, spec.class_name, spec.method_name):
            if spec.argument_index is None or len(call.arguments) <= spec.argument_index:
                continue
            argument = call.arguments[spec.argument_index]
            values = _fallback_java_m09_string_values(
                source,
                argument.text,
                argument.start,
                methods,
                frozenset(),
            )
            if not values:
                evidence.append(
                    JavaEvidence(
                        sink_name=spec.sink_name,
                        expression=argument.text,
                        label="unknown",
                        reason=(
                            "algorithm could not be resolved by the "
                            "conservative modern-Java fallback"
                        ),
                        start=argument.start,
                        end=argument.end,
                    )
                )
                continue
            for value in sorted(values):
                label, reason = (
                    _classify_java_signature_algorithm(value)
                    if spec.class_name == "Signature"
                    else _classify_java_digest_algorithm(value)
                )
                evidence.append(
                    JavaEvidence(
                        sink_name=spec.sink_name,
                        expression=argument.text,
                        label=label,
                        reason=f"{reason}; resolved by modern-Java fallback",
                        start=argument.start,
                        end=argument.end,
                    )
                )
    return evidence


def _fallback_java_m09_string_values(
    source: str,
    expression: str,
    index: int,
    methods: tuple,
    seen: frozenset[str],
) -> frozenset[str]:
    if len(seen) > 24:
        return frozenset()
    expr = strip_outer_expression(expression)
    literal = java_string_literal_value(expr)
    if literal is not None:
        return frozenset({literal})
    variable = simple_variable(expr)
    if variable is not None:
        key = f"variable:{variable}:{index}"
        if key in seen:
            return frozenset()
        assignment = _latest_java_m09_assignment(source, variable, index)
        if assignment is None:
            return frozenset()
        assigned_expression, assignment_start = assignment
        return _fallback_java_m09_string_values(
            source,
            assigned_expression,
            assignment_start,
            methods,
            seen | frozenset({key}),
        )
    helper = re.fullmatch(
        r"(?:(?:this|[A-Za-z_]\w*)\s*\.\s*)?"
        r"(?P<name>[A-Za-z_]\w*)\s*\((?P<args>.*)\)",
        expr,
        flags=re.DOTALL,
    )
    if helper is None:
        return frozenset()
    helper_name = helper.group("name")
    key = f"helper:{helper_name}"
    if key in seen:
        return frozenset()
    values: set[str] = set()
    for method in methods:
        if method.name != helper_name:
            continue
        returned = _java_m09_method_algorithm_literals(
            source,
            method.body_start,
            method.body_end,
        )
        if returned is None:
            return frozenset()
        values.update(returned)
    return frozenset(values)


def _latest_java_m09_assignment(
    source: str,
    variable: str,
    index: int,
) -> tuple[str, int] | None:
    masked = mask_java_comments_and_literals(source)
    candidates: list[tuple[int, int]] = []
    for pattern in (
        re.compile(rf"\b(?:String|var)\s+{re.escape(variable)}\s*="),
        re.compile(rf"(?<![\w$]){re.escape(variable)}\s*=(?!=)"),
    ):
        for match in pattern.finditer(masked, 0, index):
            equals = masked.find("=", match.start(), match.end())
            if equals != -1:
                candidates.append((match.start(), equals + 1))
    for assignment_start, expression_start in sorted(candidates, reverse=True):
        expression_end = _java_m09_statement_end(
            source,
            expression_start,
            index,
        )
        if expression_end != -1:
            return (
                source[expression_start:expression_end].strip(),
                assignment_start,
            )
    return None


def _java_m09_statement_end(source: str, start: int, limit: int) -> int:
    stack: list[str] = []
    quote = ""
    escaped = False
    closing = {")": "(", "]": "[", "}": "{"}
    for position in range(start, min(limit, len(source))):
        char = source[position]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char in {'"', "'"}:
            quote = char
        elif char in "([{":
            stack.append(char)
        elif char in ")]}":
            if stack and stack[-1] == closing[char]:
                stack.pop()
        elif char == ";" and not stack:
            return position
    return -1


def _java_m09_method_algorithm_literals(
    source: str,
    body_start: int,
    body_end: int,
) -> frozenset[str] | None:
    body = strip_java_comments(source[body_start + 1 : body_end])
    body_masked = mask_java_comments_and_literals(body)
    literal_pattern = r'"(?:\\.|[^"\\])*"'
    values: set[str] = set()
    if re.search(r"\breturn\s+switch\s*\(", body_masked):
        for arrow in re.finditer(r"->", body_masked):
            tail = body[arrow.end() :].lstrip()
            literal_match = re.match(literal_pattern, tail, flags=re.DOTALL)
            if literal_match is not None:
                literal = java_string_literal_value(literal_match.group(0))
                if literal is not None:
                    values.add(literal)
                continue
            if re.match(r"throw\b", tail):
                continue
            return None
        for returned in re.finditer(
            r"\breturn\b(?!\s+switch\b)",
            body_masked,
        ):
            tail = body[returned.end() :].lstrip()
            literal_match = re.match(literal_pattern, tail, flags=re.DOTALL)
            if literal_match is None:
                return None
            literal = java_string_literal_value(literal_match.group(0))
            if literal is not None:
                values.add(literal)
        return frozenset(values)
    for returned in re.finditer(r"\b(?:return|yield)\b", body_masked):
        tail = body[returned.end() :].lstrip()
        literal_match = re.match(literal_pattern, tail, flags=re.DOTALL)
        if literal_match is None:
            return None
        literal = java_string_literal_value(literal_match.group(0))
        if literal is not None:
            values.add(literal)
    return frozenset(values)


def _python_m09_algorithm_descriptor_call(
    analyzer: PythonAnalyzer,
    call: CallExpression,
) -> bool:
    """Exclude PyCA algorithm objects that do not themselves compute a digest."""

    callee = re.sub(r"\s+", "", call.callee)
    head, separator, tail = callee.partition(".")
    imported = analyzer.import_aliases.get(head)
    canonical = (
        imported + (separator + tail if separator else "")
        if imported is not None
        else callee
    )
    return re.fullmatch(
        r"(?:cryptography\.hazmat\.primitives\.)?hashes\."
        r"(?:MD5|SHA1|SHA224|SHA256|SHA384|SHA512|"
        r"SHA3_224|SHA3_256|SHA3_384|SHA3_512)",
        canonical,
    ) is not None

def _is_python_hashlib_callable(
    analyzer: SourceAnalyzer,
    call: CallExpression,
) -> bool:
    resolved = analyzer.resolve_call_target_strings(call)
    expressions = (
        call.callee,
        resolved.expression,
        *(step.expression for step in resolved.steps),
    )
    return any(
        re.search(
            r"\bhashlib\s*\.\s*(?:md5|sha1|sha224|sha256|sha384|sha512|"
            r"sha3_224|sha3_256|sha3_384|sha3_512|blake2b|blake2s)\b"
            r"|\bgetattr\s*\(\s*hashlib\s*,",
            expression,
        )
        for expression in expressions
    )



MISUSE_ID = "M09"


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
        implementation=_detect_m09,
    )


__all__ = ("MISUSE_ID", "detect")


def _csharp_selected_inline_hash_findings(
    analyzer: SourceAnalyzer,
) -> tuple[list[DetectorFinding], tuple[tuple[int, int], ...]]:
    """Resolve a fixed selector in an inline C# conditional hash factory."""

    if not isinstance(analyzer, CStyleAnalyzer):
        return [], ()
    findings: list[DetectorFinding] = []
    selected_spans: set[tuple[int, int]] = set()
    for assignment in analyzer.assignments:
        calls = tuple(
            call
            for call in analyzer.calls("Create")
            if assignment.start <= call.start < assignment.end
            and _algorithm_from_csharp_callee(call.callee)
        )
        if len(calls) < 2 or "?" not in assignment.expression:
            continue
        choices = tuple(
            re.finditer(
                r"(?P<selector>[A-Za-z_]\w*)\s*==\s*"
                r"\"(?P<value>[^\"]+)\"\s*\)\s*\?\s*"
                r"(?P<factory>(?:MD5|SHA1|SHA224|SHA256|SHA384|SHA512|"
                r"SHA3_256|SHA3_384|SHA3_512)\s*\.\s*Create\s*\(\s*\))",
                assignment.expression,
            )
        )
        if not choices:
            continue
        selector = choices[0].group("selector")
        if any(choice.group("selector") != selector for choice in choices):
            continue
        resolved_selector = analyzer.resolve_strings(selector, assignment.start)
        if len(resolved_selector.values) != 1:
            continue
        selected_value = _normalize_algorithm(resolved_selector.values[0])
        selected_algorithm = next(
            (
                _algorithm_from_csharp_callee(choice.group("factory"))
                for choice in choices
                if _normalize_algorithm(choice.group("value")) == selected_value
            ),
            "",
        )
        selected_call = next(
            (
                call
                for call in calls
                if _algorithm_from_csharp_callee(call.callee)
                == selected_algorithm
            ),
            None,
        )
        if selected_call is None:
            continue
        resolved = ResolvedValue(
            selected_algorithm,
            "constant",
            (selected_algorithm,),
        )
        findings.append(
            _algorithm_finding(
                analyzer,
                "M09",
                selected_call,
                selected_algorithm,
                resolved,
                _classify_digest,
            )
        )
        selected_spans.add((assignment.start, assignment.end))
    return findings, tuple(sorted(selected_spans))


def _csharp_runtime_inline_hash_dispatch_findings(
    analyzer: SourceAnalyzer,
) -> tuple[list[DetectorFinding], tuple[tuple[int, int], ...]]:
    """Collapse a runtime-selected inline hash branch to one unknown sink."""

    if not isinstance(analyzer, CStyleAnalyzer):
        return [], ()
    comparisons = tuple(
        re.finditer(
            r"(?:\bif|\belse\s+if)\s*\(\s*"
            r"(?P<selector>[A-Za-z_]\w*)\s*==\s*"
            r"[\"'](?P<value>[^\"']+)[\"']\s*\)",
            analyzer.source,
        )
    )
    if not comparisons:
        return [], ()

    findings: list[DetectorFinding] = []
    selected_spans: set[tuple[int, int]] = set()
    selectors = tuple(dict.fromkeys(match.group("selector") for match in comparisons))
    for selector in selectors:
        selector_matches = tuple(
            match for match in comparisons if match.group("selector") == selector
        )
        first_index = selector_matches[0].start()
        origin = analyzer.resolve_origin(selector, first_index)
        runtime_selected = origin.origin == "runtime_input" or any(
            step.result == "runtime_input" for step in origin.steps
        )
        if not runtime_selected:
            assignment = analyzer.latest_assignment(selector, first_index)
            indexed = (
                re.fullmatch(
                    r"[A-Za-z_]\w*\s*\[\s*(?P<index>.+)\s*\]",
                    assignment.expression,
                    flags=re.DOTALL,
                )
                if assignment is not None
                else None
            )
            if indexed is not None:
                index_origin = analyzer.resolve_origin(
                    indexed.group("index"),
                    assignment.start,
                )
                runtime_selected = (
                    index_origin.origin == "runtime_input"
                    or any(
                        step.result == "runtime_input"
                        for step in index_origin.steps
                    )
                )
                if runtime_selected:
                    origin = index_origin
        if not runtime_selected:
            continue
        containing = analyzer._containing_function(first_index)
        region_start = first_index
        region_end = containing.end if containing is not None else len(analyzer.source)
        calls = tuple(
            call
            for call in analyzer.calls("Create", "ComputeHash", "HashData")
            if region_start <= call.start < region_end
            and _algorithm_from_csharp_callee(call.callee)
        )
        algorithms = tuple(
            dict.fromkeys(
                _algorithm_from_csharp_callee(call.callee)
                for call in calls
                if _algorithm_from_csharp_callee(call.callee)
            )
        )
        labels = {_classify_digest(value)[0] for value in algorithms}
        if not calls or not {"secure", "misuse"} <= labels:
            continue
        resolved = ResolvedValue(
            selector,
            "runtime_input",
            algorithms,
            origin.steps,
            "one mutually exclusive hash branch is selected at runtime",
        )
        findings.append(
            _algorithm_finding(
                analyzer,
                "M09",
                calls[0],
                selector,
                resolved,
                _classify_digest,
            )
        )
        selected_spans.update((call.start, call.end) for call in calls)
    return _dedupe_findings(findings), tuple(sorted(selected_spans))


def _csharp_reflection_hash_findings(
    analyzer: SourceAnalyzer,
) -> list[DetectorFinding]:
    """Resolve fixed digest names passed through ``Type.GetType``."""

    if not isinstance(analyzer, CStyleAnalyzer):
        return []
    findings: list[DetectorFinding] = []
    for call in analyzer.calls("CreateInstance"):
        if not re.sub(r"\s+", "", call.callee).endswith(
            "Activator.CreateInstance"
        ):
            continue
        get_type = next(
            (
                candidate
                for candidate in analyzer.calls("GetType")
                if call.start <= candidate.start < call.end
                and re.sub(r"\s+", "", candidate.callee).endswith(
                    "Type.GetType"
                )
                and candidate.arguments
            ),
            None,
        )
        if get_type is None:
            continue
        type_expression = get_type.arguments[0]
        algorithms: set[str] = set()
        for literal in re.findall(
            r"(?:MD5|SHA1|SHA224|SHA256|SHA384|SHA512|"
            r"SHA3_256|SHA3_384|SHA3_512)",
            type_expression,
            flags=re.IGNORECASE,
        ):
            algorithms.add(literal)
        for reference in re.findall(r"\b[A-Za-z_]\w*\b", type_expression):
            if reference in {
                "System",
                "Security",
                "Cryptography",
            }:
                continue
            resolved_reference = analyzer.resolve_strings(reference, call.start)
            algorithms.update(
                value
                for value in resolved_reference.values
                if _classify_digest(value)[0] in {"secure", "misuse"}
            )
        if not algorithms:
            continue
        resolved = ResolvedValue(
            type_expression,
            "constant",
            tuple(sorted(algorithms)),
        )
        findings.append(
            _algorithm_finding(
                analyzer,
                "M09",
                call,
                type_expression,
                resolved,
                _classify_digest,
            )
        )
    return findings


def _python_call_is_terminal_method(
    analyzer: SourceAnalyzer,
    call: CallExpression,
) -> bool:
    expression = analyzer.source[call.start : call.end]
    return bool(
        re.search(
            rf"\.\s*{re.escape(call.simple_name)}\s*\(",
            expression,
        )
    )


def _typescript_hash_context_reaches_digest(
    analyzer: SourceAnalyzer,
    call: CallExpression,
) -> bool:
    """Require the specific Node Hash instance to reach digest()."""

    if not isinstance(analyzer, CStyleAnalyzer):
        return True
    enclosing_assignments = tuple(
        item
        for item in analyzer.assignments
        if item.start <= call.start <= item.end
    )
    assignment = (
        min(enclosing_assignments, key=lambda item: item.end - item.start)
        if enclosing_assignments
        else None
    )
    if assignment is None:
        statement_end = analyzer.source.find(";", call.end)
        if statement_end == -1:
            statement_end = min(len(analyzer.source), call.end + 500)
        if re.search(
            r"\.\s*(?:digest|end)\s*\(",
            analyzer.source[call.end:statement_end],
        ):
            return True
        return _typescript_returned_hash_reaches_digest(analyzer, call)
    if re.search(
        r"\.\s*(?:digest|end)\s*\(",
        analyzer.source[call.end:assignment.end],
    ):
        return True
    return _typescript_hash_assignment_reaches_digest(analyzer, assignment)


def _typescript_hash_assignment_reaches_digest(
    analyzer: CStyleAnalyzer,
    assignment: Assignment,
) -> bool:
    containing = analyzer._containing_function(assignment.start)
    scope_end = containing.end if containing is not None else len(analyzer.source)
    if assignment.name.startswith("this."):
        containing_classes = tuple(
            definition
            for definition in analyzer.classes
            if definition.start <= assignment.start <= definition.end
        )
        if containing_classes:
            scope_end = min(
                containing_classes,
                key=lambda definition: definition.end - definition.start,
            ).end
    blocking_assignment = next(
        (
            item.start
            for item in analyzer.assignments
            if item.name == assignment.name
            and item.start > assignment.start
            and item.start < scope_end
            and not analyzer._assignment_is_conditional(item)
        ),
        scope_end,
    )
    direct_pattern = re.compile(
        rf"(?<![\w$]){re.escape(assignment.name)}"
        rf"(?:\s*\.\s*[A-Za-z_$][A-Za-z0-9_$]*\s*\([^;]*?\))*"
        rf"\s*\.\s*(?:digest|end)\s*\("
    )
    if direct_pattern.search(
        analyzer.source,
        assignment.end,
        blocking_assignment,
    ):
        return True

    for alias in analyzer.assignments:
        if not assignment.end < alias.start < blocking_assignment:
            continue
        if re.sub(r"\s+", "", alias.expression) != assignment.name:
            continue
        alias_block = next(
            (
                item.start
                for item in analyzer.assignments
                if item.name == alias.name
                and item.start > alias.start
                and item.start < scope_end
                and not analyzer._assignment_is_conditional(item)
            ),
            scope_end,
        )
        alias_pattern = re.compile(
            rf"(?<![\w$]){re.escape(alias.name)}"
            rf"(?:\s*\.\s*[A-Za-z_$][A-Za-z0-9_$]*\s*\([^;]*?\))*"
            rf"\s*\.\s*(?:digest|end)\s*\("
        )
        if alias_pattern.search(analyzer.source, alias.end, alias_block):
            return True
    return False


def _typescript_returned_hash_reaches_digest(
    analyzer: CStyleAnalyzer,
    call: CallExpression,
) -> bool:
    """Follow a Hash returned by a local factory to its assigned consumer."""

    factory_name = analyzer.containing_function_name(call.start)
    if factory_name is None:
        return False
    definition = next(
        (
            item
            for item in analyzer.functions
            if item.name == factory_name and item.start <= call.start <= item.end
        ),
        None,
    )
    if definition is None:
        return False
    statement_start = max(
        analyzer.source.rfind(";", definition.start, call.start),
        analyzer.source.rfind("{", definition.start, call.start),
        analyzer.source.rfind("}", definition.start, call.start),
    ) + 1
    if re.search(r"\breturn\b", analyzer.source[statement_start:call.start]) is None:
        return False
    for factory_call in analyzer.calls(factory_name):
        if definition.start <= factory_call.start <= definition.end:
            continue
        assignments = tuple(
            item
            for item in analyzer.assignments
            if item.start <= factory_call.start <= item.end
        )
        if not assignments:
            continue
        assignment = min(assignments, key=lambda item: item.end - item.start)
        next_assignment = next(
            (
                item.start
                for item in analyzer.assignments
                if item.name == assignment.name and item.start > assignment.start
            ),
            len(analyzer.source),
        )
        if re.search(
            rf"(?<![\w$]){re.escape(assignment.name)}\s*\.\s*digest\s*\(",
            analyzer.source[assignment.end:next_assignment],
        ):
            return True
    return False


__all__ = ("MISUSE_ID", "detect")
