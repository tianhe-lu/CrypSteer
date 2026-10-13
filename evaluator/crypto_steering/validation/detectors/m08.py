"""M08 password-material detector."""

from __future__ import annotations

from dataclasses import dataclass
import re

from crypto_steering.validation.java.calls import (
    constructor_calls,
    receiver_method_calls,
    source_contains_any_token,
    statement_text,
)
from crypto_steering.validation.java.detection import (
    JavaEvidence,
    evidence_result,
    parse_failure_result,
)
from crypto_steering.validation.java.lightweight_ast import (
    method_containing_index,
)
from crypto_steering.validation.java.security_value_flow import SecurityValueFlow
from crypto_steering.validation.java.source import (
    CallSite,
    find_matching_delimiter,
    mask_java_comments_and_literals,
    split_arguments_with_spans,
)
from crypto_steering.validation.java.static_analysis import (
    simple_variable,
    strip_outer_expression,
)
from crypto_steering.validation.models import (
    DetectionResult,
    DetectorFinding,
    Language,
    ResolvedValue,
)
from crypto_steering.validation.source.index import (
    CStyleAnalyzer,
    JavaAnalyzer,
    SourceAnalyzer,
)
from crypto_steering.validation.detectors.apis import (
    _is_csharp_framework_pbkdf2_call,
    _python_kdf_instance_call,
    _typescript_kdf_calls,
    _is_python_hashlib_pbkdf2_call,
    _is_python_hashlib_scrypt_call,
    _python_hashlib_kdf_call_target,
    _python_password_argument,
)
from crypto_steering.validation.detectors.core import (
    _origin_finding,
    _finding_from_resolved,
    _resolved_value_is_registered_kdf_output,
    _password_label,
    _password_label_with_reachability,
    _result,
    _dedupe_findings,
    _suppress_unresolved_uncalled_helpers,
    _suppress_proven_inactive_findings,
    _preserve_definite_inactive_findings,
    _missing_argument_finding,
    _call_argument,
)
from crypto_steering.validation.detectors.engine import execute_detector
from crypto_steering.validation.detectors.hkdf import _python_hkdf_secret_findings
from crypto_steering.validation.detectors.m01 import _detect_m01


@dataclass(frozen=True)
class _JavaPasswordSink:
    kind: str
    sink_name: str
    argument_index: int
    class_name: str = ""
    method_name: str = ""
    receiver_kind: str = ""
    required_argument_count: int | None = None
    require_keystore_use: bool = False
    value_filter: str = "password_not_non_password_material"


_JAVA_PASSWORD_SINKS = (
    _JavaPasswordSink(
        "constructor",
        "new PBEKeySpec password",
        0,
        class_name="PBEKeySpec",
    ),
    _JavaPasswordSink(
        "constructor",
        "new KeyStore.PasswordProtection",
        0,
        class_name="PasswordProtection",
        require_keystore_use=True,
    ),
    _JavaPasswordSink(
        "constructor",
        "new PasswordAuthentication",
        1,
        class_name="PasswordAuthentication",
    ),
    _JavaPasswordSink(
        "receiver_method",
        "KeyStore.load",
        1,
        method_name="load",
        receiver_kind="keystore",
    ),
    _JavaPasswordSink(
        "receiver_method",
        "KeyStore.store",
        1,
        method_name="store",
        receiver_kind="keystore",
    ),
    _JavaPasswordSink(
        "receiver_method",
        "KeyStore.getKey",
        1,
        method_name="getKey",
        receiver_kind="keystore",
    ),
    _JavaPasswordSink(
        "receiver_method",
        "KeyStore.setKeyEntry",
        2,
        method_name="setKeyEntry",
        receiver_kind="keystore",
        required_argument_count=4,
    ),
    _JavaPasswordSink(
        "receiver_method",
        "KeyStore.setEntry",
        2,
        method_name="setEntry",
        receiver_kind="keystore",
        value_filter="protection_parameter_variable",
    ),
    _JavaPasswordSink(
        "receiver_method",
        "KeyStore.getEntry",
        1,
        method_name="getEntry",
        receiver_kind="keystore",
        value_filter="protection_parameter_variable",
    ),
    _JavaPasswordSink(
        "receiver_method",
        "KeyManagerFactory.init",
        1,
        method_name="init",
        receiver_kind="key_manager_factory",
    ),
)


def _detect_m08(analyzer: SourceAnalyzer, task_id: str | None, modern: int) -> DetectionResult:
    if isinstance(analyzer, JavaAnalyzer):
        return _detect_java_m08(analyzer, task_id)
    del modern
    findings: list[DetectorFinding] = []
    password_classifier = lambda resolved: _password_label_with_reachability(
        analyzer,
        resolved,
    )
    if analyzer.language == "typescript":
        for call, _canonical in _typescript_kdf_calls(
            analyzer,
            ("pbkdf2", "pbkdf2Sync", "scrypt", "scryptSync"),
        ):
            expression = _call_argument(call, 0, "password")
            if expression is None:
                findings.append(
                    _missing_argument_finding(analyzer, "M08", call, "password")
                )
            else:
                findings.append(
                    _origin_finding(
                        analyzer,
                        "M08",
                        call,
                        expression,
                        "password",
                        password_classifier,
                    )
                )
    elif analyzer.language == "csharp":
        for call in analyzer.calls("Pbkdf2", "Rfc2898DeriveBytes"):
            if not _is_csharp_framework_pbkdf2_call(analyzer, call):
                continue
            expression = _call_argument(call, 0, "password")
            if expression is None:
                findings.append(_missing_argument_finding(analyzer, "M08", call, "password"))
            else:
                findings.append(_origin_finding(analyzer, "M08", call, expression, "password", password_classifier))
        for call in analyzer.calls(
            "ExportEncryptedPkcs8PrivateKey",
            "TryExportEncryptedPkcs8PrivateKey",
            "ImportEncryptedPkcs8PrivateKey",
            "ImportFromEncryptedPem",
        ):
            expression = _call_argument(
                call,
                0,
                "password",
                "passwordBytes",
            )
            if expression is None:
                findings.append(
                    _missing_argument_finding(
                        analyzer,
                        "M08",
                        call,
                        "protected_private_key_password",
                    )
                )
            else:
                findings.append(
                    _origin_finding(
                        analyzer,
                        "M08",
                        call,
                        expression,
                        "protected_private_key_password",
                        password_classifier,
                    )
                )
    elif analyzer.language == "python":
        seen_hashlib_calls: set[tuple[int, int]] = set()
        if "PBKDF2HMAC" in analyzer.source or "Scrypt" in analyzer.source:
            for call in analyzer.calls("derive", "verify"):
                if not _python_kdf_instance_call(analyzer, call):
                    continue
                if call.arguments:
                    findings.append(_origin_finding(analyzer, "M08", call, call.arguments[0], "password", password_classifier))
        findings.extend(_python_hkdf_secret_findings(analyzer))
        for call in analyzer.calls(
            "BestAvailableEncryption", "load_pem_private_key", "load_der_private_key",
            "load_key_and_certificates", "serialize_key_and_certificates",
        ):
            expression = _python_password_argument(call)
            if expression is not None and expression.strip() != "None":
                findings.append(_origin_finding(analyzer, "M08", call, expression, "protected_container_password", password_classifier))
        for call in analyzer.calls("pbkdf2_hmac"):
            if not _is_python_hashlib_pbkdf2_call(analyzer, call):
                continue
            expression = _call_argument(call, 1, "password")
            if expression is None:
                findings.append(_missing_argument_finding(analyzer, "M08", call, "password"))
            else:
                findings.append(_origin_finding(analyzer, "M08", call, expression, "password", password_classifier))
            seen_hashlib_calls.add((call.start, call.end))
        for call in analyzer.calls("scrypt"):
            if not _is_python_hashlib_scrypt_call(analyzer, call):
                continue
            expression = _call_argument(call, 0, "password")
            if expression is None:
                findings.append(_missing_argument_finding(analyzer, "M08", call, "password"))
            else:
                findings.append(_origin_finding(analyzer, "M08", call, expression, "password", password_classifier))
            seen_hashlib_calls.add((call.start, call.end))
        for call in analyzer.calls():
            if (call.start, call.end) in seen_hashlib_calls:
                continue
            canonical = _python_hashlib_kdf_call_target(analyzer, call)
            password_index = (
                1
                if canonical == "hashlib.pbkdf2_hmac"
                else 0
                if canonical == "hashlib.scrypt"
                else None
            )
            if password_index is None:
                continue
            expression = _call_argument(call, password_index, "password")
            if expression is None:
                findings.append(
                    _missing_argument_finding(
                        analyzer,
                        "M08",
                        call,
                        "password",
                    )
                )
            else:
                findings.append(
                    _origin_finding(
                        analyzer,
                        "M08",
                        call,
                        expression,
                        "password",
                        password_classifier,
                    )
                )
    findings.extend(
        _password_derived_key_findings(
            analyzer,
            task_id,
        )
    )
    if task_id and task_id.startswith("F8") and not findings:
        # Cross-language F8 translations use PBKDF2-protected authenticated
        # envelopes when no native KeyStore analogue exists.
        return _result(analyzer, "M08", findings, "no password reaches the F8 protected-entry flow")
    lexical_findings = tuple(findings)
    findings = _suppress_proven_inactive_findings(analyzer, findings)
    findings = _suppress_unresolved_uncalled_helpers(analyzer, findings)
    findings = _preserve_definite_inactive_findings(
        analyzer,
        lexical_findings,
        findings,
    )
    return _result(
        analyzer,
        "M08",
        _dedupe_findings(findings),
        "no password-sensitive sink found",
    )


def _detect_java_m08(
    analyzer: JavaAnalyzer,
    task_id: str | None,
) -> DetectionResult:
    del task_id
    if not source_contains_any_token(
        analyzer.source,
        {
            "PBEKeySpec",
            "PasswordProtection",
            "PasswordAuthentication",
            "load",
            "store",
            "getKey",
            "setKeyEntry",
            "setEntry",
            "getEntry",
            "init",
            "keystore",
            "key_manager_factory",
        },
    ):
        return evidence_result(
            analyzer,
            "M08",
            (),
            no_sink_reason="no password-sensitive sink found",
        )
    if not analyzer.flow.ast_context.parse_success:
        return parse_failure_result(analyzer, "M08")

    evidence: list[JavaEvidence] = []
    for sink in _JAVA_PASSWORD_SINKS:
        if sink.kind == "constructor":
            calls = constructor_calls(analyzer.source, sink.class_name)
            for call in calls:
                if len(call.arguments) <= sink.argument_index:
                    continue
                if (
                    sink.require_keystore_use
                    and not _java_password_protection_reaches_keystore(
                        analyzer.source,
                        analyzer.flow,
                        call.start,
                    )
                ):
                    continue
                argument = call.arguments[sink.argument_index]
                if _java_password_sink_accepts(
                    sink,
                    argument.text,
                    analyzer.flow,
                    call.start,
                ):
                    evidence.append(
                        _java_password_evidence(analyzer, sink, argument)
                    )
            continue

        for receiver, call in receiver_method_calls(
            analyzer.source,
            sink.method_name,
        ):
            if not _java_password_receiver_matches(
                sink.receiver_kind,
                receiver,
                analyzer.flow,
                call.start,
            ):
                continue
            if (
                sink.required_argument_count is not None
                and len(call.arguments) != sink.required_argument_count
            ):
                continue
            if len(call.arguments) <= sink.argument_index:
                continue
            argument = call.arguments[sink.argument_index]
            if _java_password_sink_accepts(
                sink,
                argument.text,
                analyzer.flow,
                call.start,
            ):
                evidence.append(
                    _java_password_evidence(analyzer, sink, argument)
                )
        if sink.receiver_kind == "keystore":
            for call in _java_chained_keystore_calls(
                analyzer.source,
                sink.method_name,
            ):
                if (
                    sink.required_argument_count is not None
                    and len(call.arguments) != sink.required_argument_count
                ):
                    continue
                if len(call.arguments) <= sink.argument_index:
                    continue
                argument = call.arguments[sink.argument_index]
                if _java_password_sink_accepts(
                    sink,
                    argument.text,
                    analyzer.flow,
                    call.start,
                ):
                    evidence.append(
                        _java_password_evidence(analyzer, sink, argument)
                    )
    return evidence_result(
        analyzer,
        "M08",
        evidence,
        no_sink_reason="no password-sensitive sink found",
    )


def _java_password_evidence(
    analyzer: JavaAnalyzer,
    sink: _JavaPasswordSink,
    argument,
) -> JavaEvidence:
    classification = analyzer.flow.classify(
        argument.text,
        argument.start,
        "password",
    )
    return JavaEvidence(
        sink_name=sink.sink_name,
        expression=argument.text,
        label=classification.label,
        reason=classification.reason,
        start=argument.start,
        end=argument.end,
    )


def _java_password_sink_accepts(
    sink: _JavaPasswordSink,
    expression: str,
    flow: SecurityValueFlow,
    index: int,
) -> bool:
    if strip_outer_expression(expression) == "null":
        return False
    if sink.value_filter == "password_not_non_password_material":
        return not _java_looks_like_non_password_material(
            expression,
            flow,
            index,
        )
    if sink.value_filter == "protection_parameter_variable":
        return (
            simple_variable(expression) is not None
            and _java_looks_like_protection_parameter(
                expression,
                flow,
                index,
            )
        )
    return True


def _java_password_receiver_matches(
    receiver_kind: str,
    receiver: str,
    flow: SecurityValueFlow,
    index: int,
) -> bool:
    if receiver_kind == "keystore":
        return _java_is_keystore_receiver(receiver, flow, index)
    if receiver_kind == "key_manager_factory":
        return _java_is_key_manager_factory_receiver(receiver, flow, index)
    return True


def _java_is_keystore_receiver(
    receiver: str,
    flow: SecurityValueFlow,
    index: int,
) -> bool:
    if re.search(r"^ks$|keystore", receiver, flags=re.IGNORECASE):
        return True
    type_name = flow.variable_type(receiver, index) or ""
    if type_name.endswith("KeyStore"):
        return True
    assignment = flow.latest_assignment(receiver, index)
    return bool(
        assignment
        and re.search(
            r"\bKeyStore\s*\.\s*getInstance\s*\(",
            assignment.expression,
        )
    )


def _java_is_key_manager_factory_receiver(
    receiver: str,
    flow: SecurityValueFlow,
    index: int,
) -> bool:
    if re.search(r"keymanagerfactory|kmf", receiver, flags=re.IGNORECASE):
        return True
    type_name = flow.variable_type(receiver, index) or ""
    if type_name.endswith("KeyManagerFactory"):
        return True
    assignment = flow.latest_assignment(receiver, index)
    return bool(
        assignment
        and re.search(
            r"\bKeyManagerFactory\s*\.\s*getInstance\s*\(",
            assignment.expression,
        )
    )


def _java_chained_keystore_calls(
    source: str,
    method_name: str,
) -> tuple[CallSite, ...]:
    masked = mask_java_comments_and_literals(source)
    pattern = re.compile(
        rf"\b(?:[\w.]+\.)?KeyStore\s*\.\s*getInstance\s*\([^;]*?\)"
        rf"\s*\.\s*{re.escape(method_name)}\s*\(",
        flags=re.DOTALL,
    )
    calls: list[CallSite] = []
    for match in pattern.finditer(masked):
        open_paren = match.end() - 1
        close_paren = find_matching_delimiter(source, open_paren)
        if close_paren == -1:
            continue
        calls.append(
            CallSite(
                name=f"KeyStore.{method_name}",
                start=match.start(),
                end=close_paren + 1,
                open_paren=open_paren,
                close_paren=close_paren,
                arguments=split_arguments_with_spans(
                    source,
                    open_paren,
                    close_paren,
                ),
            )
        )
    return tuple(calls)


def _java_password_protection_reaches_keystore(
    source: str,
    flow: SecurityValueFlow,
    call_start: int,
) -> bool:
    protection_sinks = tuple(
        sink
        for sink in _JAVA_PASSWORD_SINKS
        if sink.value_filter == "protection_parameter_variable"
    )
    statement = statement_text(source, call_start)
    method_names = tuple(
        dict.fromkeys(
            sink.method_name for sink in protection_sinks if sink.method_name
        )
    )
    if method_names and re.search(
        r"\.\s*(?:"
        + "|".join(re.escape(name) for name in method_names)
        + r")\s*\(",
        statement,
    ):
        return True
    assigned_name = next(
        (
            assignment.name
            for assignment in flow.assignments
            if assignment.start <= call_start <= assignment.end
        ),
        None,
    )
    if assigned_name is None:
        return False
    method = method_containing_index(flow.methods, call_start)
    for sink in protection_sinks:
        for receiver, call in receiver_method_calls(source, sink.method_name):
            if call.start <= call_start or len(call.arguments) <= sink.argument_index:
                continue
            if method_containing_index(flow.methods, call.start) != method:
                continue
            if not _java_password_receiver_matches(
                sink.receiver_kind,
                receiver,
                flow,
                call.start,
            ):
                continue
            if simple_variable(call.arguments[sink.argument_index].text) == assigned_name:
                return True
    return False


def _java_looks_like_protection_parameter(
    expression: str,
    flow: SecurityValueFlow,
    index: int,
) -> bool:
    expr = strip_outer_expression(expression)
    if re.search(
        r"\bnew\s+(?:[\w.]+\.)?(?:KeyStore\.)?PasswordProtection\s*\(",
        expr,
    ):
        return True
    variable = simple_variable(expr)
    if variable:
        type_name = flow.variable_type(variable, index) or ""
        if type_name.endswith(("PasswordProtection", "ProtectionParameter")):
            return True
        assignment = flow.latest_assignment(variable, index)
        return bool(
            assignment
            and _java_looks_like_protection_parameter(
                assignment.expression,
                flow,
                assignment.start,
            )
        )
    return False


def _java_looks_like_non_password_material(
    expression: str,
    flow: SecurityValueFlow,
    index: int,
) -> bool:
    expr = strip_outer_expression(expression)
    if _java_looks_like_certificate_argument(expr):
        return True
    if _java_looks_like_key_material_expression(expr):
        return True
    candidate = _java_password_source_candidate(expr)
    variable = simple_variable(candidate)
    if not variable:
        return False
    if re.search(r"pass|pwd|credential", variable, flags=re.IGNORECASE):
        return False
    type_name = flow.variable_type(variable, index) or ""
    if (
        type_name
        and type_name not in {"char[]", "String"}
        and not type_name.endswith("PasswordProtection")
    ):
        return True
    if re.search(
        r"(?:bytes|digest|hash|key|secretkey|certificate|cert|chain|salt|iv|nonce)$",
        variable,
        flags=re.IGNORECASE,
    ):
        return True
    assignment = flow.latest_assignment(variable, index)
    return bool(
        assignment
        and (
            _java_looks_like_key_material_expression(assignment.expression)
            or _java_looks_like_certificate_argument(assignment.expression)
        )
    )


def _java_password_source_candidate(expression: str) -> str:
    expr = strip_outer_expression(expression)
    to_char = re.fullmatch(
        r"(?P<receiver>.+)\.\s*toCharArray\s*\(\s*\)",
        expr,
        flags=re.DOTALL,
    )
    return (
        strip_outer_expression(to_char.group("receiver"))
        if to_char is not None
        else expr
    )


def _java_looks_like_key_material_expression(expression: str) -> bool:
    return bool(
        re.search(
            r"\.\s*(?:getPublic|getPrivate|getEncoded|getAlgorithmParameters)\s*\("
            r"|\b(?:X509EncodedKeySpec|PKCS8EncodedKeySpec|RSAPrivateKeySpec|"
            r"KeyPair|PublicKey|PrivateKey|SecretKey|KeyGenerator|"
            r"KeyPairGenerator)\b"
            r"|\.digest\s*\(|\.doFinal\s*\(|Base64\s*\.\s*getDecoder"
            r"\s*\(\s*\)\s*\.\s*decode\s*\(",
            expression,
            flags=re.DOTALL,
        )
    )


def _java_looks_like_certificate_argument(expression: str) -> bool:
    return bool(
        re.search(
            r"\b(?:java\.security\.cert\.)?Certificate\s*(?:\[\s*\]|\[)"
            r"|\.getCertificate(?:Chain)?\s*\("
            r"|\.generateCertificates?\s*\("
            r"|\bCertificateFactory\b",
            expression,
            flags=re.DOTALL,
        )
    )


def _password_derived_key_findings(
    analyzer: SourceAnalyzer,
    task_id: str | None,
) -> list[DetectorFinding]:
    """Find password-shaped material deterministically converted into a key.

    M08 is not limited to registered PBKDF2 constructors.  A hardcoded
    passphrase hashed directly into an AES or HMAC key is still a hardcoded
    password use.  Reuse M01's sink-to-source evidence, then require an
    explicit password-role name so ordinary fixed raw keys remain owned by
    M01 alone.
    """

    if task_id and task_id.split("_", 1)[0] in {"F7", "F8"}:
        # Registered KDF/protected-container tasks already expose their
        # password argument directly through the ordinary M08 sinks.
        return []

    m01 = _detect_m01(analyzer, task_id, 100_000)
    records: list[DetectorFinding] = []
    role_pattern = re.compile(
        r"(?i)(?:password|passphrase|passwd|\bpwd\b|credential)"
    )
    for key_finding in m01.findings:
        if key_finding.sink_id == "hkdf_key_material_input":
            # HKDF role classification above is authoritative. Searching the
            # resolved literal text here can mistake values such as
            # "user-password-or-secret" for a password role even when the
            # program explicitly names the value as key material.
            continue
        key_value = ResolvedValue(
            key_finding.expression,
            key_finding.value_origin or "unknown",
            key_finding.resolved_values,
            key_finding.resolution_steps,
            (
                key_finding.reason
                if (key_finding.value_origin or "unknown") == "unknown"
                else None
            ),
        )
        if _resolved_value_is_registered_kdf_output(key_value):
            # Registered KDF calls expose their password argument directly in
            # the ordinary M08 sink pass above. Reclassifying the downstream
            # key duplicates that evidence and can conflate unrelated key
            # branches with password provenance.
            continue
        flow_evidence = tuple(
            step.expression
            for step in key_finding.resolution_steps
            if not step.action.startswith("resolve helper ")
        )
        if not any(role_pattern.search(item) for item in flow_evidence):
            continue
        if not isinstance(analyzer, CStyleAnalyzer):
            records.append(
                _finding_from_resolved(
                    analyzer,
                    "M08",
                    "password_derived_key",
                    key_finding.sink_name,
                    key_finding.expression,
                    key_finding.span.start,
                    key_finding.span.end,
                    key_value,
                    _password_label,
                )
            )
            continue
        role_names: set[str] = set()
        for role_assignment in analyzer.assignments:
            if not role_pattern.search(role_assignment.name):
                continue
            candidate_name = re.sub(
                r"^this\.",
                "",
                role_assignment.name,
                flags=re.IGNORECASE,
            )
            candidate_pattern = re.compile(
                rf"(?<![\w$]){re.escape(candidate_name)}(?![\w$])"
            )
            if any(candidate_pattern.search(item) for item in flow_evidence):
                role_names.add(candidate_name)
        source_record_count = 0
        for assignment in analyzer.assignments:
            simple_name = re.sub(
                r"^this\.",
                "",
                assignment.name,
                flags=re.IGNORECASE,
            )
            if simple_name not in role_names:
                continue
            assignment_value = analyzer.resolve_origin(
                assignment.expression,
                assignment.start,
            )
            records.append(
                _finding_from_resolved(
                    analyzer,
                    "M08",
                    "password_derived_key_source",
                    key_finding.sink_name,
                    assignment.expression,
                    assignment.start,
                    assignment.end,
                    assignment_value,
                    _password_label,
                )
            )
            source_record_count += 1
        if source_record_count == 0:
            records.append(
                _finding_from_resolved(
                    analyzer,
                    "M08",
                    "password_derived_key",
                    key_finding.sink_name,
                    key_finding.expression,
                    key_finding.span.start,
                    key_finding.span.end,
                    key_value,
                    _password_label,
                )
            )
    return records



MISUSE_ID = "M08"


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
        implementation=_detect_m08,
    )


__all__ = ("MISUSE_ID", "detect")
