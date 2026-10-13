"""M06 salt detector."""

from __future__ import annotations

from dataclasses import replace
import re

from crypto_steering.validation.java.calls import constructor_calls
from crypto_steering.validation.java.detection import (
    JavaEvidence,
    JavaSinkSpec,
    evidence_result,
    parse_failure_result,
    source_may_contain_policy_sink,
)
from crypto_steering.validation.java.randomness import (
    conditional_value_branches,
    has_explicit_random_origin,
)
from crypto_steering.validation.java.resolution import java_string_literal_value
from crypto_steering.validation.java.security_value_flow import (
    SourceClassification,
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
    CallExpression,
    JavaAnalyzer,
    PythonAnalyzer,
    SourceAnalyzer,
)
from crypto_steering.validation.detectors.apis import (
    _is_csharp_framework_pbkdf2_call,
    _typescript_kdf_calls,
    _is_python_hashlib_pbkdf2_call,
    _is_python_hashlib_scrypt_call,
    _python_hashlib_kdf_call_target,
)
from crypto_steering.validation.detectors.core import (
    _finding_from_resolved,
    _salt_label,
    _result,
    _suppress_unresolved_uncalled_helpers,
    _suppress_proven_inactive_findings,
    _preserve_definite_inactive_findings,
    _suppress_unresolved_verification_reuse,
    _missing_argument_finding,
    _call_argument,
    _argument_value,
)
from crypto_steering.validation.detectors.engine import execute_detector

from crypto_steering.validation.detectors.m05 import (
    _f9_designated_randomness_findings,
    _resolve_f9_random_origin,
)


_JAVA_SINKS = (
    JavaSinkSpec(
        id="pbe_key_spec_salt",
        kind="constructor",
        class_name="PBEKeySpec",
        sink_name="new PBEKeySpec salt",
        argument_index=1,
    ),
    JavaSinkSpec(
        id="pbe_parameter_spec_salt",
        kind="constructor",
        class_name="PBEParameterSpec",
        sink_name="new PBEParameterSpec salt",
        argument_index=0,
    ),
)

def _detect_m06(analyzer: SourceAnalyzer, task_id: str | None, modern: int) -> DetectionResult:
    if isinstance(analyzer, JavaAnalyzer):
        return _detect_java_m06(analyzer, task_id)
    del modern
    findings: list[DetectorFinding] = []
    if analyzer.language == "typescript":
        for call, _canonical in _typescript_kdf_calls(
            analyzer,
            ("pbkdf2", "pbkdf2Sync", "scrypt", "scryptSync"),
        ):
            expression = _call_argument(call, 1, "salt")
            if expression is None:
                findings.append(
                    _missing_argument_finding(analyzer, "M06", call, "kdf_salt")
                )
            else:
                findings.append(
                    _m06_origin_finding(analyzer, call, expression)
                )
    elif analyzer.language == "csharp":
        for call in analyzer.calls("Pbkdf2", "Rfc2898DeriveBytes"):
            if not _is_csharp_framework_pbkdf2_call(analyzer, call):
                continue
            expression = _call_argument(call, 1, "salt")
            if expression is None:
                findings.append(_missing_argument_finding(analyzer, "M06", call, "kdf_salt"))
            else:
                findings.append(_m06_origin_finding(analyzer, call, expression))
    elif analyzer.language == "python":
        seen_calls: set[tuple[int, int]] = set()
        for call in analyzer.calls("PBKDF2HMAC", "Scrypt"):
            expression = call.named_arguments.get("salt")
            if expression is None:
                findings.append(_missing_argument_finding(analyzer, "M06", call, "kdf_salt"))
            else:
                findings.append(_m06_origin_finding(analyzer, call, expression))
            seen_calls.add((call.start, call.end))
        for call in analyzer.calls("pbkdf2_hmac"):
            if not _is_python_hashlib_pbkdf2_call(analyzer, call):
                continue
            expression = _call_argument(call, 2, "salt")
            if expression is None:
                findings.append(_missing_argument_finding(analyzer, "M06", call, "kdf_salt"))
            else:
                findings.append(_m06_origin_finding(analyzer, call, expression))
            seen_calls.add((call.start, call.end))
        for call in analyzer.calls("scrypt"):
            if not _is_python_hashlib_scrypt_call(analyzer, call):
                continue
            expression = call.named_arguments.get("salt")
            if expression is None:
                findings.append(_missing_argument_finding(analyzer, "M06", call, "kdf_salt"))
            else:
                findings.append(_m06_origin_finding(analyzer, call, expression))
            seen_calls.add((call.start, call.end))
        for call in analyzer.calls():
            if (call.start, call.end) in seen_calls:
                continue
            canonical = _python_hashlib_kdf_call_target(analyzer, call)
            salt_index = (
                2
                if canonical == "hashlib.pbkdf2_hmac"
                else 1
                if canonical == "hashlib.scrypt"
                else None
            )
            if salt_index is None:
                continue
            expression = _call_argument(call, salt_index, "salt")
            if expression is None:
                findings.append(
                    _missing_argument_finding(analyzer, "M06", call, "kdf_salt")
                )
            else:
                findings.append(_m06_origin_finding(analyzer, call, expression))
    if task_id == "F9_T2":
        generated_salts = {
            re.sub(r"\s+", "", finding.expression)
            for finding in _f9_designated_randomness_findings(analyzer, "F9_T2")
            if finding.label in {"secure", "misuse"}
        }
        findings = [
            replace(
                finding,
                label="secure",
                reason_code="fresh_runtime_salt",
                reason="salt varies per operation; randomness quality is evaluated by M05",
            )
            if re.sub(r"\s+", "", finding.expression) in generated_salts
            else finding
            for finding in findings
        ]
    findings = _suppress_unresolved_verification_reuse(analyzer, findings)
    lexical_findings = tuple(findings)
    findings = _suppress_proven_inactive_findings(analyzer, findings)
    findings = _suppress_unresolved_uncalled_helpers(analyzer, findings)
    findings = _preserve_definite_inactive_findings(
        analyzer,
        lexical_findings,
        findings,
    )
    return _result(analyzer, "M06", findings, "no PBKDF2 salt sink found")


def _m06_origin_finding(
    analyzer: SourceAnalyzer,
    call: CallExpression,
    expression: str,
) -> DetectorFinding:
    expression = _argument_value(expression)
    resolved = _resolve_f9_random_origin(analyzer, expression, call.start)
    finding = _finding_from_resolved(
        analyzer,
        "M06",
        "kdf_salt",
        call.callee,
        expression,
        call.start,
        call.end,
        resolved,
        _salt_label,
    )
    if finding.label != "unknown" or not _salt_derives_from_kdf_password(
        analyzer,
        call,
        expression,
    ):
        return finding
    return replace(
        finding,
        label="misuse",
        reason_code="password_derived_salt",
        reason="salt is deterministically derived from the KDF password",
        value_origin="predictable",
    )


def _salt_derives_from_kdf_password(
    analyzer: SourceAnalyzer,
    call: CallExpression,
    salt_expression: str,
) -> bool:
    password_expressions: tuple[str, ...]
    if isinstance(analyzer, PythonAnalyzer) and call.simple_name in {
        "PBKDF2HMAC",
        "Scrypt",
    }:
        password_expressions = analyzer.associated_kdf_secret_expressions(call)
    else:
        direct = _call_argument(call, 0, "password")
        password_expressions = (direct,) if direct is not None else ()
    if not password_expressions:
        return False
    salt_expression = _argument_value(salt_expression)
    compact_salt = re.sub(r"\s+", "", salt_expression)

    salt = analyzer.resolve_origin(salt_expression, call.start)
    if salt.origin not in {"runtime_input", "unknown"} or any(
        step.result in {"secure_random", "non_csprng"}
        for step in salt.steps
    ):
        return False

    def simple_references(resolved: ResolvedValue, expression: str) -> set[str]:
        references: set[str] = set()
        for value in (expression, *(step.expression for step in resolved.steps)):
            stripped = value.strip()
            if re.fullmatch(r"(?:this\.)?[A-Za-z_$][A-Za-z0-9_$]*", stripped):
                references.add(re.sub(r"^this\.", "", stripped))
        return references

    salt_references = simple_references(salt, salt_expression)
    for raw_password_expression in password_expressions:
        password_expression = _argument_value(raw_password_expression)
        compact_password = re.sub(r"\s+", "", password_expression)
        if compact_salt == compact_password:
            return True
        salt_property = re.fullmatch(
            r"(?P<owner>(?:this\.)?[A-Za-z_$][A-Za-z0-9_$]*)\."
            r"(?P<property>[A-Za-z_$][A-Za-z0-9_$]*)",
            compact_salt,
        )
        password_property = re.fullmatch(
            r"(?P<owner>(?:this\.)?[A-Za-z_$][A-Za-z0-9_$]*)\."
            r"(?P<property>[A-Za-z_$][A-Za-z0-9_$]*)",
            compact_password,
        )
        if (
            salt_property is not None
            and password_property is not None
            and salt_property.group("owner") == password_property.group("owner")
            and salt_property.group("property") != password_property.group("property")
        ):
            continue
        password = analyzer.resolve_origin(password_expression, call.start)
        if salt_references & simple_references(password, password_expression):
            return True
    return False



def _detect_java_m06(
    analyzer: JavaAnalyzer,
    task_id: str | None,
) -> DetectionResult:
    if task_id == "F9_T2":
        return _detect_java_f9_t2_m06(analyzer)
    return _detect_java_m06_standard(analyzer)


def _detect_java_m06_standard(analyzer: JavaAnalyzer) -> DetectionResult:
    if not source_may_contain_policy_sink(analyzer.source, _JAVA_SINKS):
        return evidence_result(
            analyzer,
            "M06",
            (),
            no_sink_reason="no PBE salt-sensitive sink found",
        )
    if not analyzer.flow.ast_context.parse_success:
        return parse_failure_result(analyzer, "M06")

    evidence: list[JavaEvidence] = []
    for spec in _JAVA_SINKS:
        if spec.argument_index is None:
            continue
        for call in constructor_calls(analyzer.source, spec.class_name):
            if len(call.arguments) <= spec.argument_index:
                continue
            argument = call.arguments[spec.argument_index]
            classification = analyzer.flow.classify(
                argument.text,
                argument.start,
                "salt",
            )
            alternatives = _java_salt_assignment_alternatives(
                analyzer,
                argument.text,
                argument.start,
            )
            if alternatives is not None:
                classification = alternatives
            if (
                classification.label == "misuse"
                and _java_salt_is_runtime_provided(
                    analyzer,
                    argument.text,
                    argument.start,
                )
                and not _java_salt_has_predictable_source(
                    analyzer,
                    argument.text,
                    argument.start,
                    seen=frozenset(),
                )
            ):
                classification = replace(
                    classification,
                    label="unknown",
                    reason=(
                        "caller-provided salt is not proven freshly generated "
                        "or fixed"
                    ),
                )
            evidence.append(
                JavaEvidence(
                    sink_name=spec.sink_name,
                    expression=argument.text,
                    label=classification.label,
                    reason=classification.reason,
                    start=argument.start,
                    end=argument.end,
                )
            )
    return evidence_result(
        analyzer,
        "M06",
        evidence,
        no_sink_reason="no PBE salt-sensitive sink found",
    )


def _java_salt_assignment_alternatives(
    analyzer: JavaAnalyzer,
    expression: str,
    index: int,
) -> SourceClassification | None:
    """Combine branch assignments that can independently reach one salt."""

    variable = simple_variable(strip_outer_expression(expression))
    if variable is None:
        return None
    assignments = sorted(
        (
            assignment
            for assignment in analyzer.flow.assignments
            if assignment.name == variable
            and not assignment.type_name
            and analyzer.flow._assignment_is_code(assignment)
            and analyzer.flow._assignment_visible_at(assignment, index)
            and analyzer.position_is_reachable(assignment.start)
        ),
        key=lambda assignment: assignment.start,
    )
    if len(assignments) < 2:
        return None

    classifications: list[SourceClassification] = []
    for assignment_index, assignment in enumerate(assignments):
        assigned = analyzer.flow._assignment_expression(assignment)
        branch_end = (
            assignments[assignment_index + 1].start
            if assignment_index + 1 < len(assignments)
            else index
        )
        if _java_assignment_fills_fresh_salt(
            analyzer,
            variable,
            assigned,
            assignment.end,
            branch_end,
        ):
            classification = SourceClassification(
                "secure",
                "salt branch fills a new buffer from a varying source",
            )
        else:
            classification = analyzer.flow.classify(
                assigned,
                assignment.start,
                "salt",
            )
        if (
            _java_expression_uses_runtime_input(
                analyzer,
                assigned,
                assignment.start,
                seen=frozenset(),
            )
            and not _java_salt_has_predictable_source(
                analyzer,
                assigned,
                assignment.start,
                seen=frozenset(),
            )
        ):
            classification = SourceClassification(
                "unknown",
                "caller-provided salt freshness is unresolved",
            )
        classifications.append(classification)

    labels = {item.label for item in classifications}
    if "mixed" in labels or {"secure", "misuse"} <= labels:
        return SourceClassification(
            "mixed",
            "reachable salt assignments include fresh and predictable sources",
        )
    if "misuse" in labels:
        return SourceClassification(
            "misuse",
            "a reachable salt assignment is fixed or predictable",
        )
    if labels == {"secure"}:
        return SourceClassification(
            "secure",
            "all reachable salt assignments are freshly generated",
        )
    return SourceClassification(
        "unknown",
        "reachable salt assignments are not fully classified",
    )


def _java_assignment_fills_fresh_salt(
    analyzer: JavaAnalyzer,
    variable: str,
    expression: str,
    after: int,
    before: int,
) -> bool:
    """Recognize a branch-local buffer populated before another branch wins."""

    if re.fullmatch(
        r"new\s+(?:byte|char)\s*\[[^]]+\]",
        strip_outer_expression(expression),
        flags=re.DOTALL,
    ) is None:
        return False
    return analyzer.flow._variable_filled_by_secure_random(
        variable,
        before,
        after=after,
    ) or analyzer.flow._variable_filled_by_varying_salt_source(
        variable,
        before,
        after=after,
    )


def _java_salt_has_predictable_source(
    analyzer: JavaAnalyzer,
    expression: str,
    index: int,
    *,
    seen: frozenset[tuple[str, int]],
) -> bool:
    """Find a definite fixed branch without treating output buffers as salt."""

    value = strip_outer_expression(expression)
    key = (value, index)
    if not value or key in seen:
        return False
    next_seen = seen | frozenset({key})
    if analyzer.flow._is_predictable_salt_source(value):
        return True
    if java_string_literal_value(value) is not None or re.fullmatch(
        r"(?:new\s+)?(?:byte|char)\s*(?:\[\s*\])?\s*\{[^}]*\}",
        value,
        flags=re.DOTALL,
    ):
        return True

    branches = conditional_value_branches(value)
    if branches is not None:
        return any(
            _java_salt_has_predictable_source(
                analyzer,
                branch,
                index,
                seen=next_seen,
            )
            for branch in branches
        )

    transparent = re.fullmatch(
        r"(?P<source>.+)\.\s*(?:getBytes|toCharArray|clone)\s*\([^)]*\)",
        value,
        flags=re.DOTALL,
    )
    if transparent is not None:
        return _java_salt_has_predictable_source(
            analyzer,
            transparent.group("source"),
            index,
            seen=next_seen,
        )

    allocation = re.fullmatch(
        r"new\s+(?:byte|char)\s*\[[^]]+\]",
        value,
        flags=re.DOTALL,
    )
    if allocation is not None:
        return True

    variable = simple_variable(value)
    if variable is not None:
        assignments = [
            assignment
            for assignment in analyzer.flow.assignments
            if assignment.name == variable
            and analyzer.flow._assignment_is_code(assignment)
            and analyzer.flow._assignment_visible_at(assignment, index)
            and analyzer.position_is_reachable(assignment.start)
        ]
        for assignment in assignments:
            assigned = analyzer.flow._assignment_expression(assignment)
            if re.fullmatch(
                r"new\s+(?:byte|char)\s*\[[^]]+\]",
                strip_outer_expression(assigned),
                flags=re.DOTALL,
            ) and _java_array_has_runtime_writes(
                analyzer,
                variable,
                assignment.end,
                index,
            ):
                continue
            if _java_salt_has_predictable_source(
                analyzer,
                assigned,
                assignment.start,
                seen=next_seen,
            ):
                return True
        containing = next(
            (
                method
                for method in analyzer.flow.methods
                if method.body_start < index < method.body_end
                and any(
                    parameter.name == variable
                    for parameter in method.params
                )
            ),
            None,
        )
        if containing is not None:
            parameter_names = [
                parameter.name for parameter in containing.params
            ]
            parameter_index = parameter_names.index(variable)
            for call in analyzer.flow._method_calls(containing.name):
                if not analyzer.flow._call_may_target_declared_method(
                    call,
                    containing,
                ):
                    continue
                if containing.start <= call.start <= containing.body_end:
                    continue
                if len(call.arguments) != len(parameter_names):
                    continue
                argument = call.arguments[parameter_index]
                if _java_salt_has_predictable_source(
                    analyzer,
                    argument.text,
                    argument.start,
                    seen=next_seen,
                ):
                    return True
        return False

    helper = re.fullmatch(
        r"(?:[A-Za-z_]\w*\s*\.\s*)?(?P<name>[A-Za-z_]\w*)\s*\(.*\)",
        value,
        flags=re.DOTALL,
    )
    if helper is None:
        return False
    for method in analyzer.flow.methods:
        if method.name != helper.group("name"):
            continue
        body = analyzer.source[method.body_start + 1 : method.body_end]
        for returned in re.finditer(r"\breturn\s+(?P<value>[^;]+);", body):
            return_index = method.body_start + 1 + returned.start("value")
            if _java_salt_has_predictable_source(
                analyzer,
                returned.group("value"),
                return_index,
                seen=next_seen,
            ):
                return True
    return False


def _java_array_has_runtime_writes(
    analyzer: JavaAnalyzer,
    variable: str,
    after: int,
    before: int,
) -> bool:
    for match in re.finditer(
        rf"\b{re.escape(variable)}\s*\[[^\]\r\n]+\]\s*=\s*"
        r"(?P<value>[^;]+);",
        analyzer.source[after:before],
    ):
        value_start = after + match.start("value")
        if _java_expression_uses_runtime_input(
            analyzer,
            match.group("value"),
            value_start,
            seen=frozenset(),
        ):
            return True
    return False


def _java_salt_is_runtime_provided(
    analyzer: JavaAnalyzer,
    expression: str,
    index: int,
) -> bool:
    """Recognize caller/user salt data hidden by deterministic decoding.

    Runtime-provided salt is not proof of either freshness or reuse, so M06
    reports it as unknown. This includes input decoded by a local helper and
    byte arrays populated element-by-element from an input string.
    """

    if _java_expression_uses_runtime_input(
        analyzer,
        expression,
        index,
        seen=frozenset(),
    ):
        return True

    variable = re.fullmatch(r"[A-Za-z_]\w*", expression.strip())
    if variable is None:
        return False
    name = re.escape(variable.group(0))
    for match in re.finditer(
        rf"\b{name}\s*\[[^\]\r\n]+\]\s*=\s*(?P<value>[^;]+);",
        analyzer.source[:index],
    ):
        if _java_expression_uses_runtime_input(
            analyzer,
            match.group("value"),
            match.start("value"),
            seen=frozenset(),
        ):
            return True
    return False


def _java_expression_uses_runtime_input(
    analyzer: JavaAnalyzer,
    expression: str,
    index: int,
    *,
    seen: frozenset[tuple[str, int]],
) -> bool:
    value = expression.strip()
    key = (value, index)
    if not value or key in seen:
        return False
    next_seen = seen | frozenset({key})
    if re.search(
        r"\b(?:System\s*\.\s*)?console\s*\(\s*\)\s*\.\s*"
        r"(?:readLine|readPassword)\s*\(|"
        r"\b(?:System\s*\.\s*)?getenv\s*\(|"
        r"\bargs\s*\[|"
        r"\bnew\s+(?:java\.util\.)?Scanner\s*\(\s*System\s*\.\s*in\s*\)"
        r"\s*\.\s*(?:next|nextLine)\s*\(",
        value,
    ):
        return True
    for call in re.finditer(
        r"\b(?P<receiver>[A-Za-z_]\w*)\s*\.\s*"
        r"(?:next|nextLine|readLine|readPassword)\s*\(",
        value,
    ):
        receiver_assignment = analyzer.flow._latest_assignment(
            call.group("receiver"),
            index,
        )
        if receiver_assignment is None:
            continue
        receiver_source = analyzer.flow._assignment_expression(
            receiver_assignment
        )
        if re.search(
            r"\b(?:System\s*\.\s*)?in\b|"
            r"\b(?:System\s*\.\s*)?console\s*\(\s*\)|"
            r"\bConsole\s*\.\s*in\b",
            receiver_source,
            flags=re.IGNORECASE,
        ):
            return True
    for name in dict.fromkeys(re.findall(r"\b[A-Za-z_]\w*\b", value)):
        assignment = analyzer.flow._latest_assignment(name, index)
        if assignment is None:
            containing = next(
                (
                    method
                    for method in analyzer.flow.methods
                    if method.body_start < index < method.body_end
                    and any(parameter.name == name for parameter in method.params)
                ),
                None,
            )
            if containing is None:
                continue
            parameter_names = [parameter.name for parameter in containing.params]
            parameter_index = parameter_names.index(name)
            for call in analyzer.flow._method_calls(containing.name):
                if not analyzer.flow._call_may_target_declared_method(
                    call,
                    containing,
                ):
                    continue
                if containing.start <= call.start <= containing.body_end:
                    continue
                if len(call.arguments) != len(parameter_names):
                    continue
                argument = call.arguments[parameter_index]
                if _java_expression_uses_runtime_input(
                    analyzer,
                    argument.text,
                    argument.start,
                    seen=next_seen,
                ):
                    return True
            continue
        if _java_expression_uses_runtime_input(
            analyzer,
            analyzer.flow._assignment_expression(assignment),
            assignment.start,
            seen=next_seen,
        ):
            return True
    return False


def _detect_java_f9_t2_m06(analyzer: JavaAnalyzer) -> DetectionResult:
    """Treat a fresh per-operation salt as non-M06 evidence.

    F9_T2 varies generator quality for M05. A general-purpose PRNG remains a
    fresh salt source, while M05 reports that generator as insecure.
    """

    evidence: list[JavaEvidence] = []
    for class_name, argument_index in (
        ("PBEKeySpec", 1),
        ("PBEParameterSpec", 0),
    ):
        for call in constructor_calls(analyzer.source, class_name):
            if len(call.arguments) <= argument_index:
                continue
            argument = call.arguments[argument_index]
            if not has_explicit_random_origin(
                analyzer.flow,
                argument.text,
                argument.start,
            ):
                return _detect_java_m06_standard(analyzer)
            evidence.append(
                JavaEvidence(
                    sink_name=f"new {class_name} salt",
                    sink_id="new_pbekeyspec_salt",
                    expression=argument.text,
                    label="secure",
                    reason_code="fresh_runtime_salt",
                    reason=(
                        "salt varies per operation; randomness quality is "
                        "evaluated by M05"
                    ),
                    policy_id="M06.policy.v1",
                    start=argument.start,
                    end=argument.end,
                )
            )
    if not evidence:
        return _detect_java_m06_standard(analyzer)
    return evidence_result(
        analyzer,
        "M06",
        evidence,
        no_sink_reason="no PBKDF2 salt sink found",
        preserve_filtered_summary=False,
        apply_reachability=False,
    )


MISUSE_ID = "M06"


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
        implementation=_detect_m06,
    )


__all__ = ("MISUSE_ID", "detect")
