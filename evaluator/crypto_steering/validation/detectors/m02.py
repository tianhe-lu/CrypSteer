"""M02 cipher-mode detector."""

from __future__ import annotations

import re

from crypto_steering.validation.java.detection import (
    JavaEvidence,
    JavaSinkSpec,
    evidence_result,
    iter_policy_arguments,
    source_may_contain_policy_sink,
)
from crypto_steering.validation.java.resolution import (
    normalize_algorithm as _normalize_java_algorithm,
    split_transformation,
)
from crypto_steering.validation.models import (
    DetectionResult,
    DetectorFinding,
    Language,
    ProgramLabel,
    ResolvedValue,
)
from crypto_steering.validation.source.index import (
    CStyleAnalyzer,
    CallExpression,
    JavaAnalyzer,
    PythonAnalyzer,
    SourceAnalyzer,
)
from crypto_steering.validation.detectors.apis import (
    _has_csharp_symmetric_terminal,
)
from crypto_steering.validation.detectors.core import (
    _algorithm_finding,
    _classify_ts_mode,
    _with_typescript_cipher_mode_fragments,
    _classify_csharp_mode,
    _classify_python_mode,
    _python_mode_constructor_target,
    _result,
    _suppress_proven_inactive_findings,
    _missing_argument_finding,
    _direct_call_finding,
    _direct_policy_finding,
)
from crypto_steering.validation.detectors.engine import execute_detector


_JAVA_SINKS = (
    JavaSinkSpec(
        id="cipher_get_instance",
        kind="static_method",
        class_name="Cipher",
        method_name="getInstance",
        sink_name="Cipher.getInstance transformation",
        argument_index=0,
    ),
)
_JAVA_INSECURE_ALGORITHMS = {
    "DES",
    "DESEDE",
    "TRIPLEDES",
    "BLOWFISH",
    "RC2",
    "RC4",
    "ARCFOUR",
}
_JAVA_INSECURE_MODES = {"ECB"}
_JAVA_BARE_MISUSE = {"AES", "DES", "DESEDE", "TRIPLEDES", "BLOWFISH"}
_JAVA_SECURE_MODES = {
    "GCM",
    "CCM",
    "CTR",
    "CBC",
    "CFB",
    "OFB",
    "CHACHA20POLY1305",
}
_JAVA_SUPPORTED_PADDINGS = {"NOPADDING", "PKCS5PADDING", "ISO10126PADDING"}


def _detect_m02(analyzer: SourceAnalyzer, task_id: str | None, modern: int) -> DetectionResult:
    if isinstance(analyzer, JavaAnalyzer):
        return _detect_java_m02(analyzer, task_id)
    del task_id, modern
    findings: list[DetectorFinding] = []
    preserve_library_modes = False
    if analyzer.language == "typescript":
        for call in analyzer.calls("createCipheriv", "createDecipheriv"):
            if not call.arguments:
                findings.append(_missing_argument_finding(analyzer, "M02", call, "cipher_algorithm"))
                continue
            resolved = analyzer.resolve_strings(call.arguments[0], call.start)
            resolved = _with_typescript_cipher_mode_fragments(resolved)
            findings.append(_algorithm_finding(analyzer, "M02", call, call.arguments[0], resolved, _classify_ts_mode))
    elif analyzer.language == "csharp":
        terminal_mode_findings: list[DetectorFinding] = []
        library_mode_findings: list[DetectorFinding] = []
        for call in analyzer.calls("CreateEncryptor", "CreateDecryptor"):
            compact = re.sub(r"\s+", "", call.callee)
            if "." not in compact:
                continue
            receiver = compact.rsplit(".", 1)[0]
            expression = f"{receiver}.Mode"
            resolved = analyzer.resolve_strings(expression, call.start)
            if resolved.origin == "unknown" and not resolved.values:
                continue
            terminal_mode_findings.append(
                _algorithm_finding(
                    analyzer,
                    "M02",
                    call,
                    expression,
                    resolved,
                    _classify_csharp_mode,
                )
            )
            containing = (
                analyzer._containing_function(call.start)
                if isinstance(analyzer, CStyleAnalyzer)
                else None
            )
            if (
                containing is not None
                and not analyzer.containing_function_is_called(call.start)
            ):
                receiver_assignments = [
                    assignment
                    for assignment in analyzer.assignments
                    if assignment.name == expression
                    and analyzer._assignment_scopes[assignment] == containing
                ]
                if len(receiver_assignments) > 1:
                    library_mode_findings.extend(
                        _algorithm_finding(
                            analyzer,
                            "M02",
                            CallExpression(
                                "SymmetricAlgorithm.Mode",
                                (assignment.expression,),
                                {},
                                assignment.start,
                                assignment.end,
                            ),
                            assignment.expression,
                            analyzer.resolve_strings(
                                assignment.expression,
                                assignment.start,
                            ),
                            _classify_csharp_mode,
                        )
                        for assignment in receiver_assignments
                    )
        if isinstance(analyzer, CStyleAnalyzer):
            for assignment in analyzer.assignments:
                if assignment.name.rsplit(".", 1)[-1] != "Mode":
                    continue
                if (
                    "." not in assignment.name
                    and re.search(r"\bCipherMode\s*\.", assignment.expression)
                    is None
                ):
                    # Bare ``Mode`` members also occur in unrelated object
                    # initializers such as FileStreamOptions.  A symmetric
                    # mode assignment either has an explicit receiver
                    # (``aes.Mode``) or names a CipherMode value.
                    continue
                resolved = analyzer.resolve_strings(assignment.expression, assignment.start)
                synthetic = CallExpression("SymmetricAlgorithm.Mode", (assignment.expression,), {}, assignment.start, assignment.end)
                findings.append(_algorithm_finding(analyzer, "M02", synthetic, assignment.expression, resolved, _classify_csharp_mode))
        if library_mode_findings:
            findings = library_mode_findings
            preserve_library_modes = True
        elif terminal_mode_findings:
            findings = terminal_mode_findings
        if not findings and _has_csharp_symmetric_terminal(analyzer.source):
            if re.search(r"\bAesGcm\b|\bAesCcm\b", analyzer.source):
                findings.append(_direct_policy_finding(analyzer, "M02", "aead_primitive", "AesGcm/AesCcm", "secure", "approved_aead", "explicit AEAD primitive is secure"))
            elif re.search(
                r"\bAes\s*\.\s*Create\s*\("
                r"|\bnew\s+(?:AesManaged|AesCryptoServiceProvider|RijndaelManaged)\s*\(",
                analyzer.source,
            ):
                findings.append(_direct_policy_finding(analyzer, "M02", "aes_default_mode", "Aes.Create default", "secure", "secure_platform_default", "Aes uses the documented CBC default when Mode is not assigned"))
    elif analyzer.language == "python":
        for call in analyzer.calls("Cipher"):
            if len(call.arguments) < 2:
                findings.append(
                    _missing_argument_finding(
                        analyzer,
                        "M02",
                        call,
                        "cipher_mode",
                    )
                )
                continue
            expression = call.arguments[1]
            if isinstance(analyzer, PythonAnalyzer):
                resolved = analyzer.resolve_callable_strings(
                    expression,
                    call.start,
                )
            else:
                target = _python_mode_constructor_target(expression)
                if target is None:
                    resolved = ResolvedValue(
                        expression,
                        "unknown",
                        unresolved_reason=(
                            "cipher mode expression is not a constructor call"
                        ),
                    )
                else:
                    resolved = analyzer.resolve_strings(target, call.start)
            findings.append(
                _algorithm_finding(
                    analyzer,
                    "M02",
                    call,
                    expression,
                    resolved,
                    _classify_python_mode,
                )
            )
        for call in analyzer.calls("AESGCM", "AESCCM"):
            findings.append(_direct_call_finding(analyzer, "M02", call, "secure", "approved_aead", f"{call.simple_name} is an approved AEAD primitive"))
        if analyzer.calls("encrypt", "decrypt"):
            for call in analyzer.calls("Fernet"):
                findings.append(
                    _direct_call_finding(
                        analyzer,
                        "M02",
                        call,
                        "secure",
                        "approved_authenticated_primitive",
                        "Fernet provides authenticated symmetric encryption",
                    )
                )
    if not preserve_library_modes:
        findings = _suppress_proven_inactive_findings(analyzer, findings)
    return _result(analyzer, "M02", findings, "no symmetric cipher mode sink found")


def _detect_java_m02(
    analyzer: JavaAnalyzer,
    task_id: str | None,
) -> DetectionResult:
    del task_id
    if not source_may_contain_policy_sink(analyzer.source, _JAVA_SINKS):
        return evidence_result(
            analyzer,
            "M02",
            (),
            no_sink_reason="no AES cipher-mode choice found",
        )

    evidence: list[JavaEvidence] = []
    for argument in iter_policy_arguments(analyzer.source, _JAVA_SINKS):
        transformations = analyzer.string_values(
            argument.expression,
            argument.start,
        )
        if transformations is None:
            label: ProgramLabel = "unknown"
            reason = (
                "cipher transformation could not be resolved to a finite "
                "set of strings"
            )
        else:
            symmetric = tuple(
                value
                for value in sorted(transformations)
                if _java_m02_transformation_is_in_scope(value)
            )
            if not symmetric:
                continue
            label, reason = _classify_java_cipher_transformations(symmetric)
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
        "M02",
        evidence,
        no_sink_reason="no AES cipher-mode choice found",
        preserve_filtered_summary=False,
    )


def _classify_java_cipher_transformations(
    transformations: tuple[str, ...],
) -> tuple[ProgramLabel, str]:
    classifications = [
        _classify_java_cipher_transformation(transformation)
        for transformation in transformations
    ]
    labels = {label for label, _reason in classifications}
    reasons = "; ".join(
        dict.fromkeys(reason for _label, reason in classifications)
    )
    if "mixed" in labels or {"secure", "misuse"} <= labels:
        return "mixed", reasons
    if "misuse" in labels:
        return "misuse", reasons
    if labels == {"secure"}:
        return "secure", reasons
    return "unknown", reasons


def _classify_java_cipher_transformation(
    transformation: str,
) -> tuple[ProgramLabel, str]:
    algorithm, mode, padding = split_transformation(transformation)
    normalized_algorithm = _normalize_java_algorithm(algorithm)
    normalized_mode = _normalize_java_algorithm(mode or "")
    if _java_transformation_uses_weak_algorithm(normalized_algorithm):
        return (
            "misuse",
            f"cipher transformation uses weak algorithm {transformation!r}",
        )
    parts = [part.strip() for part in transformation.split("/")]
    if len(parts) not in {1, 2, 3} or any(not part for part in parts):
        return "unknown", f"cipher transformation {transformation!r} is malformed"
    if (
        padding is not None
        and _normalize_java_algorithm(padding)
        in _JAVA_INSECURE_MODES | _JAVA_SECURE_MODES
    ):
        return (
            "unknown",
            f"cipher transformation {transformation!r} uses a mode name as padding",
        )
    if _normalize_java_algorithm(transformation) in _JAVA_SECURE_MODES:
        return (
            "secure",
            f"cipher transformation {transformation!r} uses an explicit AEAD mode",
        )
    if normalized_algorithm in {"RSA", "ECIES"}:
        return (
            "unknown",
            f"asymmetric transformation {transformation!r} is outside M02 symmetric-mode policy",
        )
    if mode is None and normalized_algorithm in _JAVA_BARE_MISUSE:
        return (
            "misuse",
            f"bare cipher transformation {transformation!r} uses provider default mode",
        )
    if normalized_mode in _JAVA_INSECURE_MODES:
        return "misuse", f"cipher transformation uses insecure mode {mode!r}"
    if normalized_mode in _JAVA_SECURE_MODES:
        if (
            padding is not None
            and _normalize_java_algorithm(padding)
            not in _JAVA_SUPPORTED_PADDINGS
        ):
            return (
                "secure",
                f"cipher transformation uses explicit non-ECB mode {mode!r}; "
                "padding compatibility is outside the M02 policy label",
            )
        return (
            "secure",
            f"cipher transformation uses explicit non-ECB mode {mode!r}",
        )
    return (
        "unknown",
        f"cipher transformation {transformation!r} has unclassified mode",
    )


def _java_transformation_uses_weak_algorithm(
    normalized_algorithm: str,
) -> bool:
    if normalized_algorithm in _JAVA_INSECURE_ALGORITHMS:
        return True
    if not normalized_algorithm.startswith("PBE"):
        return False
    return any(
        weak in normalized_algorithm
        for weak in ("DES", "BLOWFISH", "RC2", "RC4", "ARCFOUR")
    )


def _java_m02_transformation_is_in_scope(transformation: str) -> bool:
    normalized = _normalize_java_algorithm(
        split_transformation(transformation)[0]
    )
    if normalized in {"RSA", "ECIES"}:
        return False
    # Modern password-based constructions are evaluated by the KDF-related
    # categories. Keep only PBE names whose embedded legacy cipher is itself
    # definite M02 misuse evidence.
    return not normalized.startswith("PBE") or (
        _java_transformation_uses_weak_algorithm(normalized)
    )



MISUSE_ID = "M02"


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
        implementation=_detect_m02,
    )


__all__ = ("MISUSE_ID", "detect")
