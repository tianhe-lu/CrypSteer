"""M01 key-material detector."""

from __future__ import annotations

from dataclasses import replace
import re

from crypto_steering.validation.java.calls import (
    constructor_calls,
    receiver_method_calls,
    source_contains_any_token,
    static_method_calls,
)
from crypto_steering.validation.java.detection import (
    JavaEvidence,
    evidence_result,
    parse_failure_result,
)
from crypto_steering.validation.java.resolution import (
    normalize_algorithm as _normalize_java_algorithm,
    resolve_string_expression,
    split_transformation,
)
from crypto_steering.validation.java.security_value_flow import (
    SecurityValueFlow,
    security_context_present,
)
from crypto_steering.validation.models import (
    DetectionResult,
    DetectorFinding,
    Language,
    ProgramLabel,
    ResolutionStep,
    ResolvedValue,
)
from crypto_steering.validation.source.index import (
    CStyleAnalyzer,
    Assignment,
    CallExpression,
    JavaAnalyzer,
    SourceAnalyzer,
    csharp_activator_constructor_arguments,
)
from crypto_steering.validation.detectors.core import (
    _origin_call_findings,
    _origin_finding,
    _finding_from_resolved,
    _resolved_is_explicit_mac_algorithm,
    _m01_key_label,
    _result,
    _suppress_unresolved_internal_adapters,
    _suppress_unresolved_uncalled_helpers,
    _python_m01_internal_kdf_hmac_finding,
    _suppress_proven_inactive_findings,
    _preserve_definite_inactive_findings,
    _direct_call_finding,
    _csharp_object_initializer_property,
    _call_argument,
    _python_canonical_detector_call_name,
)
from crypto_steering.validation.detectors.engine import execute_detector
from crypto_steering.validation.detectors.apis import _is_typescript_crypto_function_call
from crypto_steering.validation.detectors.hkdf import _python_hkdf_key_material_findings

from crypto_steering.validation.detectors.m05 import _f9_mutated_buffer_origin


def _detect_m01(analyzer: SourceAnalyzer, task_id: str | None, modern: int) -> DetectionResult:
    if isinstance(analyzer, JavaAnalyzer):
        return _detect_java_m01(analyzer, task_id)
    del modern
    kdf_task = bool(task_id and task_id.split("_", 1)[0] in {"F7", "F8"})

    def key_classifier(
        resolved: ResolvedValue,
    ) -> tuple[ProgramLabel, str, str]:
        # A byte-array allocation becomes secure key material after a CSPRNG
        # fill. Preserve that mutation provenance when the value crosses a
        # local-function boundary and the generic resolver retains only the
        # allocation expression.
        if analyzer.language in {"typescript", "csharp"}:
            mutation_index = max(
                (step.span.start for step in resolved.steps),
                default=len(analyzer.source),
            )
            mutated_origin = _f9_mutated_buffer_origin(
                analyzer,
                resolved.expression,
                mutation_index,
            )
            if mutated_origin is not None:
                resolved = replace(resolved, origin=mutated_origin)
        if kdf_task and _resolved_value_calls_local_registered_kdf(
            analyzer,
            resolved,
        ):
            return (
                "secure",
                "registered_kdf_derived_key",
                "key is produced by a registered KDF; KDF arguments are evaluated by M06-M08",
            )
        return _m01_key_label(resolved, kdf_task=kdf_task)

    findings: list[DetectorFinding] = []
    if analyzer.language == "typescript":
        specs = (
            ("createCipheriv", 1),
            ("createDecipheriv", 1),
            ("createHmac", 1),
            ("createIncrementalHmac", 1),
            ("createNodeHmac", 1),
        )
        for name, argument_index in specs:
            for call in analyzer.calls(name):
                if (
                    name in {"createCipheriv", "createDecipheriv", "createHmac"}
                    and not _is_typescript_crypto_function_call(
                        analyzer,
                        call,
                        {name},
                    )
                ):
                    # A local wrapper may deliberately share the native API's
                    # name. Its argument positions belong to the wrapper, not
                    # Node's API contract; the native call inside is the sink.
                    continue
                expression = _call_argument(call, argument_index)
                if expression is None:
                    continue
                findings.append(
                    _origin_finding(
                        analyzer,
                        "M01",
                        call,
                        expression,
                        "cryptographic_key",
                        key_classifier,
                    )
                )
    elif analyzer.language == "csharp":
        specs = (
            ("AesGcm", 0), ("AesCcm", 0), ("HMACMD5", 0), ("HMACSHA1", 0),
            ("HMACSHA256", 0), ("HMACSHA384", 0), ("HMACSHA512", 0),
            ("HMACSHA3_256", 0), ("HMACSHA3_384", 0), ("HMACSHA3_512", 0),
            ("CreateHmac", 1), ("CreateIncrementalHmac", 1),
            ("CreateNativeHmac", 1), ("CreateHMAC", 1),
        )
        findings.extend(_origin_call_findings(analyzer, "M01", specs, "cryptographic_key", key_classifier))
        for call in analyzer.calls("CreateInstance"):
            compact = re.sub(r"\s+", "", call.callee)
            if not compact.endswith("Activator.CreateInstance") or len(call.arguments) < 2:
                continue
            algorithm_type = analyzer.resolve_strings(call.arguments[0], call.start)
            if not _resolved_is_explicit_mac_algorithm(algorithm_type):
                continue
            constructor_arguments = csharp_activator_constructor_arguments(
                analyzer.source[call.start : call.end]
            )
            if not constructor_arguments:
                continue
            findings.append(
                _origin_finding(
                    analyzer,
                    "M01",
                    call,
                    constructor_arguments[0],
                    "hmac_key",
                    key_classifier,
                )
            )
        for call in analyzer.calls(
            "HMACMD5", "HMACSHA1", "HMACSHA256", "HMACSHA384", "HMACSHA512",
            "HMACSHA3_256", "HMACSHA3_384", "HMACSHA3_512",
        ):
            if not call.arguments:
                initializer_key = _csharp_object_initializer_property(
                    analyzer.source[call.start : call.end],
                    "Key",
                )
                if initializer_key is not None:
                    findings.append(
                        _origin_finding(
                            analyzer,
                            "M01",
                            call,
                            initializer_key,
                            "hmac_key",
                            key_classifier,
                        )
                    )
                    continue
                findings.append(
                    _direct_call_finding(
                        analyzer,
                        "M01",
                        call,
                        "secure",
                        "provider_generated_hmac_key",
                        "zero-argument HMAC construction generates provider key material",
                    )
                )
        for call in analyzer.calls("CreateEncryptor", "CreateDecryptor"):
            if len(call.arguments) >= 2:
                findings.append(_origin_finding(analyzer, "M01", call, call.arguments[0], "aes_key", key_classifier))
            elif not call.arguments:
                receiver = re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0]
                if re.fullmatch(r"[A-Za-z_$][\w$]*", receiver):
                    key_expression = f"{receiver}.Key"
                    caller_resolved = _csharp_zero_arg_transform_key_from_callers(
                        analyzer,
                        call,
                        receiver,
                    )
                    assignment = analyzer.latest_assignment(
                        key_expression,
                        call.start,
                    )
                    if caller_resolved is not None:
                        findings.append(
                            _finding_from_resolved(
                                analyzer,
                                "M01",
                                "aes_key",
                                call.callee,
                                key_expression,
                                call.start,
                                call.end,
                                caller_resolved,
                                key_classifier,
                            )
                        )
                    elif assignment is not None and isinstance(analyzer, CStyleAnalyzer):
                        resolved = _csharp_key_property_value_at_first_use(
                            analyzer,
                            assignment,
                        )
                        findings.append(
                            _finding_from_resolved(
                                analyzer,
                                "M01",
                                "aes_key",
                                call.callee,
                                key_expression,
                                call.start,
                                call.end,
                                resolved,
                                key_classifier,
                            )
                        )
                    else:
                        findings.append(
                            _origin_finding(
                                analyzer,
                                "M01",
                                call,
                                key_expression,
                                "aes_key",
                                key_classifier,
                            )
                        )
                elif not re.search(r"\.\s*Key\s*=", analyzer.source):
                    findings.append(
                        _direct_call_finding(
                            analyzer,
                            "M01",
                            call,
                            "secure",
                            "provider_generated_symmetric_key",
                            "zero-argument transform creation uses the provider-generated algorithm key",
                        )
                    )
        for call in analyzer.calls("HashData"):
            if "HMAC" in call.callee.upper() and call.arguments:
                findings.append(
                    _origin_finding(
                        analyzer,
                        "M01",
                        call,
                        call.arguments[0],
                        "hmac_key",
                        key_classifier,
                    )
                )
        if isinstance(analyzer, CStyleAnalyzer):
            for assignment in analyzer.assignments:
                if (
                    "." in assignment.name
                    and assignment.name.rsplit(".", 1)[-1] == "Key"
                ):
                    if _csharp_symmetric_key_property_assignment_is_redundant(
                        analyzer,
                        assignment,
                    ):
                        continue
                    resolved = _csharp_key_property_value_at_first_use(
                        analyzer,
                        assignment,
                    )
                    findings.append(
                        _finding_from_resolved(
                            analyzer, "M01", "aes_key_property", "SymmetricAlgorithm.Key",
                            assignment.expression, assignment.start, assignment.end, resolved, key_classifier,
                        )
                    )
    elif analyzer.language == "python":
        native_key_constructors = {
            "cryptography.fernet.Fernet",
            "cryptography.hazmat.primitives.ciphers.aead.AESCCM",
            "cryptography.hazmat.primitives.ciphers.aead.AESGCM",
            "cryptography.hazmat.primitives.ciphers.algorithms.AES",
        }
        for constructor in ("AES", "AESGCM", "AESCCM", "Fernet"):
            for call in analyzer.calls(constructor):
                if (
                    _python_canonical_detector_call_name(analyzer, call)
                    not in native_key_constructors
                ):
                    continue
                expression = _call_argument(call, 0)
                if expression is None:
                    continue
                findings.append(
                    _origin_finding(
                        analyzer,
                        "M01",
                        call,
                        expression,
                        "cryptographic_key",
                        key_classifier,
                    )
                )
        for call in analyzer.calls("HMAC"):
            argument_index = (
                1
                if call.callee.endswith("hashes.HMAC")
                or call.callee == "hashes.HMAC"
                else 0
            )
            expression = _call_argument(call, argument_index)
            if expression is None:
                continue
            findings.append(
                _origin_finding(
                    analyzer,
                    "M01",
                    call,
                    expression,
                    "cryptographic_key",
                    key_classifier,
                )
            )
        findings.extend(_python_hkdf_key_material_findings(analyzer))
        for call in analyzer.calls("new"):
            if call.callee == "hmac.new" and call.arguments:
                findings.append(
                    _origin_finding(
                        analyzer,
                        "M01",
                        call,
                        call.arguments[0],
                        "cryptographic_key",
                        key_classifier,
                    )
                )
        for call in analyzer.calls("digest"):
            if call.callee == "hmac.digest" and call.arguments:
                findings.append(
                    _origin_finding(
                        analyzer,
                        "M01",
                        call,
                        call.arguments[0],
                        "hmac_key",
                        key_classifier,
                    )
                )
        findings = [
            finding
            for finding in findings
            if not _python_m01_internal_kdf_hmac_finding(analyzer, finding)
        ]
    if analyzer.language == "typescript":
        findings = _reclassify_typescript_serialized_m01_keys(findings)
        findings = _suppress_unresolved_internal_adapters(
            findings,
            adapter_names={"createNodeHmac"},
        )
    elif analyzer.language == "csharp":
        findings = _suppress_unresolved_internal_adapters(
            findings,
            adapter_names={"CreateNativeHmac", "CreateHMAC"},
        )
    elif analyzer.language == "python":
        findings = _reclassify_python_ciphertext_derived_m01_keys(findings)
    lexical_findings = tuple(findings)
    findings = _suppress_proven_inactive_findings(analyzer, findings)
    findings = _suppress_unresolved_uncalled_helpers(analyzer, findings)
    findings = _preserve_definite_inactive_findings(
        analyzer,
        lexical_findings,
        findings,
    )
    return _result(analyzer, "M01", findings, "no cryptographic key sink found")


def _detect_java_m01(
    analyzer: JavaAnalyzer,
    task_id: str | None,
) -> DetectionResult:
    del task_id
    if not source_contains_any_token(
        analyzer.source,
        {
            "SecretKeySpec",
            "DESKeySpec",
            "DESedeKeySpec",
            "Mac",
            "Cipher",
            "init",
        },
    ):
        return evidence_result(
            analyzer,
            "M01",
            (),
            no_sink_reason="no key-material sink found",
        )
    if not analyzer.flow.ast_context.parse_success:
        return parse_failure_result(analyzer, "M01")

    operation_sinks: list[JavaEvidence] = []
    constructor_sinks: list[JavaEvidence] = []
    for class_name, sink_name in (
        ("SecretKeySpec", "new SecretKeySpec"),
        ("DESKeySpec", "new DESKeySpec"),
        ("DESedeKeySpec", "new DESedeKeySpec"),
    ):
        for call in constructor_calls(analyzer.source, class_name):
            if not call.arguments:
                continue
            if _java_constructor_key_is_reassigned_before_use(
                analyzer.source,
                analyzer.flow,
                call.start,
                call.end,
            ):
                continue
            argument = call.arguments[0]
            constructor_sinks.append(
                _java_m01_origin_evidence(
                    analyzer,
                    sink_name,
                    argument.text,
                    argument.start,
                    argument.end,
                )
            )

    for factory_class, argument_index, sink_name in (
        ("Mac", 0, "Mac.init"),
        ("Cipher", 1, "Cipher.init key"),
    ):
        if not security_context_present(analyzer.source):
            continue
        receivers = _java_receivers_assigned_from(
            analyzer.source,
            factory_class,
        )
        for receiver, call in receiver_method_calls(analyzer.source, "init"):
            if receiver not in receivers or len(call.arguments) <= argument_index:
                continue
            transformation = _java_receiver_transformation_at(
                analyzer.source,
                analyzer.flow,
                factory_class,
                receiver,
                call.start,
            )
            if (
                factory_class == "Cipher"
                and transformation is not None
                and _normalize_java_algorithm(
                    split_transformation(transformation)[0]
                )
                in {"RSA", "ECIES"}
            ):
                continue
            argument = call.arguments[argument_index]
            operation_sinks.append(
                _java_m01_origin_evidence(
                    analyzer,
                    sink_name,
                    argument.text,
                    argument.start,
                    argument.end,
                )
            )

    return evidence_result(
        analyzer,
        "M01",
        operation_sinks or constructor_sinks,
        no_sink_reason="no key-material sink found",
    )


def _java_m01_origin_evidence(
    analyzer: JavaAnalyzer,
    sink_name: str,
    expression: str,
    start: int,
    end: int,
) -> JavaEvidence:
    classification = analyzer.flow.classify(
        expression,
        start,
        "key_material",
    )
    if _java_m01_has_external_keystore_origin(
        analyzer,
        expression,
        start,
        seen=frozenset(),
    ):
        classification = replace(
            classification,
            label="unknown",
            reason="key material is loaded from an external KeyStore",
        )
    return JavaEvidence(
        sink_name=sink_name,
        expression=expression,
        label=classification.label,
        reason=classification.reason,
        start=start,
        end=end,
    )


def _java_m01_has_external_keystore_origin(
    analyzer: JavaAnalyzer,
    expression: str,
    index: int,
    *,
    seen: frozenset[tuple[str, int]],
) -> bool:
    """Trace a wrapped key back to a KeyStore loaded from external data."""

    value = expression.strip()
    key = (value, index)
    if not value or key in seen:
        return False
    next_seen = seen | frozenset({key})

    get_key = re.search(
        r"\b(?P<receiver>[A-Za-z_]\w*)\s*\.\s*getKey\s*\(",
        value,
    )
    if get_key is not None and _java_keystore_loads_external_data(
        analyzer,
        get_key.group("receiver"),
        index,
    ):
        return True

    for name in dict.fromkeys(re.findall(r"\b[A-Za-z_]\w*\b", value)):
        assignment = analyzer.flow._latest_assignment(name, index)
        if assignment is None:
            continue
        if _java_m01_has_external_keystore_origin(
            analyzer,
            analyzer.flow._assignment_expression(assignment),
            assignment.start,
            seen=next_seen,
        ):
            return True
    return False


def _java_keystore_loads_external_data(
    analyzer: JavaAnalyzer,
    receiver: str,
    index: int,
) -> bool:
    receiver_type = (analyzer.flow.variable_type(receiver, index) or "").replace(
        " ",
        "",
    ).rsplit(".", 1)[-1]
    if receiver_type != "KeyStore":
        return False
    return any(
        call.start < index
        and call.arguments
        and call.arguments[0].text.strip() != "null"
        for candidate, call in receiver_method_calls(analyzer.source, "load")
        if candidate == receiver
    )


def _java_receivers_assigned_from(source: str, factory_class: str) -> set[str]:
    escaped = re.escape(factory_class)
    return {
        match.group("name")
        for match in re.finditer(
            rf"(?:\b{escaped}\s+)?(?P<name>[A-Za-z_]\w*)\s*=\s*"
            rf"(?:[\w.]+\.)?{escaped}\s*\.\s*getInstance\s*\(",
            source,
        )
    }


def _java_receiver_transformation_at(
    source: str,
    flow: SecurityValueFlow,
    factory_class: str,
    receiver: str,
    index: int,
) -> str | None:
    candidates: list[tuple[int, str]] = []
    for call in static_method_calls(source, factory_class, "getInstance"):
        if call.start >= index or not call.arguments:
            continue
        assignment = next(
            (
                item
                for item in flow.assignments
                if item.name == receiver
                and item.start <= call.start <= item.end
            ),
            None,
        )
        if assignment is None:
            continue
        resolved = resolve_string_expression(
            call.arguments[0].text,
            flow,
            call.arguments[0].start,
        )
        if resolved is not None:
            candidates.append((call.start, resolved))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _java_constructor_key_is_reassigned_before_use(
    source: str,
    flow: SecurityValueFlow,
    call_start: int,
    call_end: int,
) -> bool:
    assignment = next(
        (
            item
            for item in flow.assignments
            if item.start <= call_start <= item.end and item.expression
        ),
        None,
    )
    if assignment is None:
        return False
    next_assignment = _java_next_assignment(
        source,
        assignment.name,
        call_end,
    )
    if next_assignment == -1:
        return False
    next_use = _java_next_security_init_use(
        source,
        assignment.name,
        call_end,
    )
    return next_use == -1 or next_assignment < next_use


def _java_next_assignment(source: str, variable: str, start: int) -> int:
    match = re.search(
        rf"(?<![.\w$]){re.escape(variable)}\s*=(?!=)",
        source[start:],
    )
    return -1 if match is None else start + match.start()


def _java_next_security_init_use(
    source: str,
    variable: str,
    start: int,
) -> int:
    match = re.search(
        rf"\.\s*init\s*\([^;]*\b{re.escape(variable)}\b[^;]*\)",
        source[start:],
        flags=re.DOTALL,
    )
    return -1 if match is None else start + match.start()



MISUSE_ID = "M01"


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
        implementation=_detect_m01,
    )


__all__ = ("MISUSE_ID", "detect")


def _csharp_zero_arg_transform_key_from_callers(
    analyzer: SourceAnalyzer,
    call: CallExpression,
    receiver: str,
) -> ResolvedValue | None:
    if not isinstance(analyzer, CStyleAnalyzer):
        return None
    containing = min(
        (
            function
            for function in analyzer.functions
            if function.start <= call.start <= function.end
        ),
        key=lambda function: function.end - function.start,
        default=None,
    )
    if containing is None or receiver not in containing.parameters:
        return None
    parameter_index = containing.parameters.index(receiver)
    overload_arities = {
        len(definition.parameters)
        for definition in analyzer.function(containing.name)
    }
    values: list[ResolvedValue] = []
    for caller in analyzer.calls(containing.name):
        if containing.start <= caller.start <= containing.end:
            continue
        if not analyzer._call_can_target_function(caller, containing):
            continue
        if (
            len(overload_arities) > 1
            and len(caller.arguments) != len(containing.parameters)
        ):
            continue
        argument = analyzer._c_style_argument_for_parameter(
            caller,
            parameter_index,
            receiver,
        )
        if argument is None:
            continue
        compact_argument = re.sub(r"\s+", "", argument)
        if re.fullmatch(r"[A-Za-z_]\w*", compact_argument):
            provider_value = analyzer._resolve_csharp_provider_property(
                f"{compact_argument}.Key",
                caller.start,
                mode="origin",
                seen=frozenset(),
                bindings=None,
            )
            if provider_value is None:
                continue
            values.append(provider_value)
        else:
            if re.search(
                r"(?:Aes|AES|DES|TripleDES|Rijndael|SymmetricAlgorithm)"
                r"(?:\s*\.\s*Create\s*\(|\s*\()",
                argument,
            ) is None:
                continue
            values.append(
                analyzer.resolve_origin(f"{argument}.Key", caller.start)
            )
    if not values:
        return None
    origins = {value.origin for value in values}
    if len(origins) == 1:
        origin = values[0].origin
        reason = None
    else:
        origin = "unknown"
        reason = "caller receiver keys have multiple incompatible origins"
    return ResolvedValue(
        expression=f"{receiver}.Key",
        origin=origin,
        values=tuple(
            sorted({item for value in values for item in value.values})
        ),
        steps=tuple(step for value in values for step in value.steps),
        unresolved_reason=reason,
    )


def _reclassify_typescript_serialized_m01_keys(
    findings: list[DetectorFinding],
) -> list[DetectorFinding]:
    """Resolve key fields recovered from a source-defined encrypted record.

    Generic origin resolution deliberately treats a slice of a heterogeneous
    ``Buffer.concat`` record as unresolved.  For M01 we can use the other
    active key sinks as a stronger data-flow reference: a recovered field is
    secure when its trace reaches the exact fresh-key producer used by a
    secure sink.  A selected field that instead reaches only cipher output is
    an unapproved ciphertext-derived key and therefore misuse.
    """

    secure_producers = {
        _m01_provenance_identity(step)
        for finding in findings
        if finding.label == "secure"
        for step in finding.resolution_steps
        if _m01_step_has_fresh_provenance(step)
        and _m01_provenance_identity(step) is not None
    }
    updated: list[DetectorFinding] = []
    for finding in findings:
        if finding.label != "unknown" or not _m01_uses_serialized_field(finding):
            updated.append(finding)
            continue
        selected_producers = {
            identity
            for step in finding.resolution_steps
            if _m01_step_has_fresh_provenance(step)
            and (identity := _m01_provenance_identity(step)) is not None
        }
        if (
            secure_producers & selected_producers
            and _typescript_serialized_access_selects_fresh_key(finding)
        ):
            updated.append(
                replace(
                    finding,
                    label="secure",
                    reason_code="recovered_generated_key",
                    reason=(
                        "serialized key suffix reaches the same fresh key "
                        "producer as an active cryptographic sink"
                    ),
                )
            )
            continue
        if any(
            re.search(
                r"(?:"
                r"\b(?:cipher(?:text)?|encryptor|decryptor|operation)\s*\.\s*"
                r"(?:update|final)"
                r"|\bcreateCipheriv\s*\([^)]*\)\s*\.\s*(?:update|final)"
                r")\s*\(",
                step.expression,
                flags=re.IGNORECASE,
            )
            for step in finding.resolution_steps
        ):
            updated.append(
                replace(
                    finding,
                    label="misuse",
                    reason_code="ciphertext_derived_key",
                    reason=(
                        "serialized field used as a key includes cipher output "
                        "rather than approved key generation"
                    ),
                    value_origin="predictable",
                )
            )
            continue
        if secure_producers & selected_producers:
            updated.append(
                replace(
                    finding,
                    label="secure",
                    reason_code="recovered_generated_key",
                    reason=(
                        "serialized key field reaches the same fresh key "
                        "producer as an active cryptographic sink"
                    ),
                )
            )
            continue
        updated.append(finding)
    return updated


def _typescript_serialized_access_selects_fresh_key(
    finding: DetectorFinding,
) -> bool:
    """Recognize common layouts that explicitly serialize a generated key.

    A heterogeneous record also contains IV and ciphertext origins, so generic
    origin joining cannot identify the selected field.  These bounded layouts
    establish that the selected prefix, suffix, or first delimited field is
    the key.  Interior slices remain unresolved and can be classified as
    ciphertext-derived by the caller.
    """

    expressions = (finding.expression,) + tuple(
        step.expression for step in finding.resolution_steps
    )
    if any(
        re.search(
            r"\.\s*(?:slice|subarray)\s*\(\s*-\s*\d+\s*\)",
            expression,
        )
        for expression in expressions
    ):
        return True
    has_prefix_slice = any(
        re.search(
            r"\.\s*(?:slice|subarray)\s*\(\s*0\s*,\s*\d+\s*\)",
            expression,
        )
        for expression in expressions
    )
    record_starts_with_key = any(
        re.search(
            r"(?:Buffer\s*\.\s*concat\s*\(\s*\[|\[)\s*"
            r"(?:key|secret|keyBytes|keyMaterial)\b",
            expression,
            flags=re.IGNORECASE,
        )
        for expression in expressions
    )
    if has_prefix_slice and record_starts_with_key:
        return True
    has_first_delimited_field = any(
        re.search(r"\.\s*split\s*\([^)]*\)\s*\[\s*0\s*\]", expression)
        for expression in expressions
    )
    serialized_key_first = any(
        re.search(
            r"^\s*(?:key|secret|keyBytes|keyMaterial)\b"
            r"[^+]*\.\s*toString\s*\([^)]*\)\s*\+",
            expression,
            flags=re.IGNORECASE,
        )
        for expression in expressions
    )
    return has_first_delimited_field and serialized_key_first


def _reclassify_python_ciphertext_derived_m01_keys(
    findings: list[DetectorFinding],
) -> list[DetectorFinding]:
    """Resolve Python key fields recovered from a local encrypted record."""

    secure_producers = {
        identity
        for finding in findings
        if finding.label == "secure"
        for step in finding.resolution_steps
        if _m01_step_has_fresh_provenance(step)
        and (identity := _m01_provenance_identity(step)) is not None
    }
    updated: list[DetectorFinding] = []
    for finding in findings:
        expressions = (finding.expression,) + tuple(
            step.expression for step in finding.resolution_steps
        )
        if finding.label != "unknown" or not any(
            re.search(r"\[[^]]*:[^]]*\]", expression)
            for expression in expressions
        ):
            updated.append(finding)
            continue
        selected_producers = {
            identity
            for step in finding.resolution_steps
            if _m01_step_has_fresh_provenance(step)
            and (identity := _m01_provenance_identity(step)) is not None
        }
        if secure_producers & selected_producers:
            updated.append(
                replace(
                    finding,
                    label="secure",
                    reason_code="recovered_generated_key",
                    reason=(
                        "serialized key field reaches the same fresh key "
                        "producer as an active cryptographic sink"
                    ),
                    value_origin="derived_secure",
                )
            )
            continue
        if not any(
            re.search(
                r"(?:"
                r"\b(?:encryptor|decryptor|cipher(?:text)?|operation)\s*\.\s*"
                r"(?:encrypt|decrypt|update|finalize)"
                r"|\b(?:AESGCM|AESCCM|ChaCha20Poly1305|Fernet|Cipher)\s*"
                r"\([^)]*\)\s*\.\s*(?:encrypt|decrypt)"
                r")\s*\(",
                step.expression,
                flags=re.IGNORECASE,
            )
            for step in finding.resolution_steps
        ):
            updated.append(finding)
            continue
        updated.append(
            replace(
                finding,
                label="misuse",
                reason_code="ciphertext_derived_key",
                reason=(
                    "serialized field used as a key includes cipher output "
                    "rather than approved key generation"
                ),
                value_origin="predictable",
            )
        )
    return updated


def _m01_provenance_identity(step: ResolutionStep) -> tuple[str, str] | None:
    expression = re.sub(r"\s+", "", step.expression)
    if re.search(
        r"(?:^|\.)(?:randomBytes|randomFill|randomFillSync|getRandomValues)\(",
        expression,
    ):
        return ("fresh-api", expression)
    if step.span.start < 0:
        return None
    return (str(step.span.start), expression)


def _m01_step_has_fresh_provenance(step: ResolutionStep) -> bool:
    if step.result in {"secure_random", "derived_secure"}:
        return True
    expression = re.sub(r"\s+", "", step.expression)
    return bool(
        re.search(
            r"(?:"
            r"(?:^|\.)(?:randomBytes|randomFill|randomFillSync|getRandomValues)\("
            r"|\bos\s*\.\s*urandom\s*\("
            r"|\bsecrets\s*\.\s*(?:token_bytes|randbits|randbelow)\s*\("
            r"|\b(?:AESGCM|Fernet)\s*\.\s*generate_key\s*\("
            r")",
            expression,
        )
    )


def _m01_uses_serialized_field(finding: DetectorFinding) -> bool:
    expressions = (finding.expression,) + tuple(
        step.expression for step in finding.resolution_steps
    )
    return any(
        re.search(
            r"\.\s*(?:slice|subarray|split)\s*\(|"
            r"\b(?:readFile|readFileSync|ReadAllBytes|ReadAllText)\s*\(",
            expression,
        )
        for expression in expressions
    )


def _csharp_key_property_value_at_first_use(
    analyzer: CStyleAnalyzer,
    assignment: Assignment,
) -> ResolvedValue:
    """Resolve a property after any fill that precedes its first key use."""

    containing = min(
        (
            function
            for function in analyzer.functions
            if function.start <= assignment.start <= function.end
        ),
        key=lambda function: function.end - function.start,
        default=None,
    )
    scope_end = containing.end if containing is not None else len(analyzer.source)
    first_use = min(
        (
            call.start
            for call in analyzer.calls(
                "CreateEncryptor",
                "CreateDecryptor",
                "TransformFinalBlock",
                "Encrypt",
                "Decrypt",
                "EncryptCbc",
                "DecryptCbc",
            )
            if assignment.end < call.start <= scope_end
        ),
        default=scope_end,
    )
    if "." not in assignment.name:
        mutation = analyzer.latest_mutation_origin(
            assignment.name,
            first_use,
            seen=frozenset(),
            bindings=None,
        )
        if mutation is not None and mutation[0].start > assignment.end:
            return mutation[1]
    return analyzer.resolve_origin(assignment.expression, assignment.start)


def _csharp_symmetric_key_property_assignment_is_redundant(
    analyzer: CStyleAnalyzer,
    assignment: Assignment,
) -> bool:
    """Let transform calls, rather than provider setup, own symmetric keys."""

    receiver = assignment.name.rsplit(".", 1)[0]
    receiver_assignment = analyzer.latest_assignment(receiver, assignment.start)
    initializer = (
        receiver_assignment.expression if receiver_assignment is not None else ""
    )
    if re.search(
        r"\b(?:Aes|DES|TripleDES|Rijndael|SymmetricAlgorithm)\b",
        initializer,
    ):
        return True
    prefix = analyzer.source[max(0, assignment.start - 240) : assignment.start]
    return bool(
        re.search(
            rf"\b(?:Aes|DES|TripleDES|Rijndael|SymmetricAlgorithm)\b"
            rf"[^;{{}}]*\b{re.escape(receiver)}\b",
            prefix,
        )
    )


def _resolved_value_calls_local_registered_kdf(
    analyzer: SourceAnalyzer,
    resolved: ResolvedValue,
) -> bool:
    """Recognize a local wrapper whose returned value comes from a native KDF."""

    if not isinstance(analyzer, CStyleAnalyzer):
        return False
    expressions = (resolved.expression,) + tuple(
        step.expression for step in resolved.steps
    )
    helper_names = {
        match.group("name")
        for expression in expressions
        if (
            match := re.match(
                r"\s*(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*\(",
                expression,
            )
        )
    }
    kdf_pattern = re.compile(
        r"\b(?:pbkdf2|pbkdf2Sync|pbkdf2_hmac|Pbkdf2|scrypt|scryptSync)\s*\(|"
        r"\bPBKDF2HMAC\s*\(|"
        r"\bRfc2898DeriveBytes\b",
        flags=re.IGNORECASE,
    )
    return any(
        kdf_pattern.search(analyzer.source[definition.start : definition.end])
        for name in helper_names
        for definition in analyzer.function(name)
    )


__all__ = ("MISUSE_ID", "detect")
