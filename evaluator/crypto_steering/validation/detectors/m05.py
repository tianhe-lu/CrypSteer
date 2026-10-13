"""M05 security-sensitive-randomness detector."""

from __future__ import annotations

import ast
from dataclasses import replace
import re

from crypto_steering.validation.java.calls import (
    constructor_calls,
    method_calls,
    source_contains_any_token,
    static_method_calls,
)
from crypto_steering.validation.java.detection import (
    JavaEvidence,
    evidence_result,
    parse_failure_result,
)
from crypto_steering.validation.java.randomness import (
    has_explicit_random_origin,
    random_origin_evidence,
)
from crypto_steering.validation.java.security_value_flow import (
    SecurityValueFlow,
    SourceClassification,
    security_context_present,
)
from crypto_steering.validation.models import (
    DetectionResult,
    DetectorFinding,
    Language,
    ProgramLabel,
    ResolvedValue,
    ResolutionStep,
    SourceSpan,
    ValueOrigin,
    combine_detector_findings,
)
from crypto_steering.validation.source.index import (
    CStyleAnalyzer,
    CallExpression,
    FunctionDefinition,
    JavaAnalyzer,
    PythonAnalyzer,
    SourceAnalyzer,
    split_arguments,
)
from crypto_steering.validation.detectors.apis import (
    _security_sensitive_randomness_findings,
    _resolved_value_has_random_source_evidence,
    _resolved_value_has_any_random_source_evidence,
    _randomness_sink_values,
    _is_random_source_call,
    _random_source_api_origin,
)
from crypto_steering.validation.detectors.core import (
    _finding_from_resolved,
    _matching_c_style_brace,
    _matching_c_style_parenthesis,
    _random_label,
    _result,
    _dedupe_findings,
    _suppress_custom_prng_seed_findings,
    _suppress_proven_inactive_findings,
    _preserve_definite_inactive_findings,
    _call_argument,
    _argument_value,
    _line_column,
)
from crypto_steering.validation.detectors.engine import (
    execute_detector,
)


_F9_RANDOM_OUTPUT_ROLE = re.compile(
    r"(?i)(?:"
    r"challenge[_$]?(?:bytes|buffer|value|integer|number|int|code|id)(?:[_$]?value)?|"
    r"nonce[_$]?(?:value|integer|number|int|code|id)(?:[_$]?value)?|"
    r"random[_$]?(?:integer|number|value|int)|"
    r"secure[_$]?random[_$]?(?:integer|number|value|int)|"
    r"(?:(?:one[_$]?time|candidate|generated)[_$]?)?"
    r"challenge(?:bytes|buffer|value|integer|number|int|code|id)?|"
    r"nonce(?:value|integer|number|int|code|id)?|"
    r"random(?:integer|number|value|int)|"
    r"secure[_$]?random(?:integer|number|value|int)|"
    r"operation[_$]?value|security[_$]?code|"
    r"pin(?:value|integer|number|int|code)?|"
    r"otp(?:value|integer|number|int|code)?|"
    r"(?:numeric|verification|access|authentication|auth)[_$]?code|"
    r"(?:(?:one[_$]?time|otp|challenge)[_$]?)?code(?:value|integer|number|int)?|"
    r"token(?:value|integer|number|int)?"
    r")"
)
_F9_STRONG_INTEGER_OUTPUT_ROLE = re.compile(
    r"(?i)(?:"
    r"challenge[_$]?(?:value|integer|number|int|code|id)(?:[_$]?value)?|"
    r"nonce[_$]?(?:value|integer|number|int|code|id)(?:[_$]?value)?|"
    r"random[_$]?(?:integer|number|value|int)|"
    r"secure[_$]?random[_$]?(?:integer|number|value|int)|"
    r"(?:(?:one[_$]?time|candidate|generated)[_$]?)?"
    r"challenge(?:value|integer|number|int|code|id)?|"
    r"nonce(?:value|integer|number|int|code|id)?|"
    r"random(?:integer|number|value|int)|"
    r"secure[_$]?random(?:integer|number|value|int)|"
    r"operation[_$]?value|security[_$]?code|"
    r"pin(?:value|integer|number|int|code)?|"
    r"otp(?:value|integer|number|int|code)?|"
    r"(?:numeric|verification|access|authentication|auth)[_$]?code|"
    r"(?:(?:one[_$]?time|otp|challenge)[_$]?)?code(?:value|integer|number|int)?|"
    r"token(?:value|integer|number|int|code)"
    r")"
)



def _detect_m05(analyzer: SourceAnalyzer, task_id: str | None, modern: int) -> DetectionResult:
    if isinstance(analyzer, JavaAnalyzer):
        return _detect_java_m05(analyzer, task_id)
    del modern
    if task_id not in {"F9_T1", "F9_T2", "F9_T3"}:
        lexical_findings = tuple(
            _security_sensitive_randomness_findings(analyzer)
        )
        findings = _suppress_proven_inactive_findings(
            analyzer,
            lexical_findings,
        )
        findings = _preserve_definite_inactive_findings(
            analyzer,
            lexical_findings,
            findings,
        )
        return _result(
            analyzer,
            "M05",
            findings,
            "no generated randomness reaches a registered security-sensitive sink",
        )

    if task_id in {"F9_T1", "F9_T2", "F9_T3"}:
        lexical_findings = tuple(
            _f9_designated_randomness_findings(analyzer, task_id)
        )
        findings = list(lexical_findings)
        if task_id == "F9_T3":
            findings = _prioritize_f9_t3_output_relevance(analyzer, findings)
        lexical_findings = tuple(findings)
        findings = _suppress_proven_inactive_findings(analyzer, findings)
        findings = _preserve_called_f9_findings(
            analyzer,
            lexical_findings,
            findings,
        )
        return _result(
            analyzer,
            "M05",
            _dedupe_findings(findings),
            "no generated randomness reaches the task-designated security sink",
        )

    raise AssertionError("unreachable M05 task dispatch")


def _preserve_called_f9_findings(
    analyzer: SourceAnalyzer,
    lexical_findings: tuple[DetectorFinding, ...],
    active_findings: list[DetectorFinding],
) -> list[DetectorFinding]:
    """Retain definite F9 evidence in a locally reachable called helper."""

    if isinstance(analyzer, PythonAnalyzer):
        # Python's analyzer already performs call-target-sensitive retention.
        # A same-named method on another class is not proof that a dormant
        # helper is invoked.
        return _dedupe_findings(active_findings)

    records = list(active_findings)
    signatures = {
        (item.span.start, item.span.end, item.expression, item.label)
        for item in records
    }
    records.extend(
        item
        for item in lexical_findings
        if item.label in {"secure", "misuse", "mixed"}
        and analyzer.containing_function_is_called(item.span.start)
        and analyzer.position_is_locally_reachable(item.span.start)
        and (item.span.start, item.span.end, item.expression, item.label)
        not in signatures
    )
    return _dedupe_findings(records)


def _f9_designated_randomness_findings(
    analyzer: SourceAnalyzer,
    task_id: str,
) -> list[DetectorFinding]:
    """Resolve only the IV, salt, or challenge governed by one F9 task."""

    desired_sink = {
        "F9_T1": "crypto_iv_or_nonce_input",
        "F9_T2": "kdf_salt_input",
    }.get(task_id)
    findings: list[DetectorFinding] = []
    if desired_sink is not None:
        desired_values = [
            value
            for value in _randomness_sink_values(analyzer)
            if value[0] == desired_sink
            and not (
                task_id == "F9_T1"
                and re.search(r"(?i)(?:decrypt|decipher)", value[1])
            )
        ]
        for sink_id, sink_name, expression, start, end in desired_values:
            resolved = _resolve_f9_random_origin(analyzer, expression, start)
            if resolved.origin not in {
                "secure_random",
                "derived_secure",
                "non_csprng",
                "predictable",
            }:
                if not _resolved_value_has_any_random_source_evidence(
                    analyzer,
                    resolved,
                ):
                    continue
            if (
                resolved.origin == "predictable"
                and not _resolved_value_has_any_random_source_evidence(
                    analyzer,
                    resolved,
                )
                and not any(
                    step.action == "resolve selected callable alternative"
                    and step.result in {"non_csprng", "predictable"}
                    for step in resolved.steps
                )
            ):
                continue
            findings.append(
                _finding_from_resolved(
                    analyzer,
                    "M05",
                    sink_id,
                    sink_name,
                    expression,
                    start,
                    end,
                    resolved,
                    _random_label,
                )
            )
        # A designated native sink must never fall back to every random API in
        # the file: encryption keys and auxiliary token material are not the
        # IV/salt controlled by F9_T1/F9_T2.  The legacy seed-oriented contract
        # is used only when the source has no native designated sink at all.
        if desired_values:
            return _dedupe_findings(findings)
        return _legacy_f9_random_source_findings(analyzer)

    findings = _f9_integer_generator_findings(analyzer)
    if findings:
        return findings

    for assignment in getattr(analyzer, "assignments", ()):
        if not (
            _f9_assignment_is_integer_role(analyzer, assignment)
            or _f9_assignment_is_byte_role(analyzer, assignment)
        ):
            continue
        resolved = _resolve_f9_random_origin(
            analyzer,
            assignment.expression,
            assignment.start,
        )
        if resolved.origin not in {"secure_random", "derived_secure", "non_csprng", "predictable"}:
            continue
        if (
            resolved.origin == "predictable"
            and not _resolved_value_has_random_source_evidence(analyzer, resolved)
        ):
            continue
        findings.append(
            _finding_from_resolved(
                analyzer,
                "M05",
                "security_random_output_role",
                assignment.name,
                assignment.expression,
                assignment.start,
                assignment.end,
                resolved,
                _random_label,
            )
        )

    if findings:
        return _prioritize_f9_output_findings(_dedupe_findings(findings))

    candidates = {
        "typescript": (
            "randomBytes", "randomFill", "randomFillSync", "randomInt",
            "getRandomValues", "random", "imul", "m05CreateBytes",
            "m05RandomSource",
        ),
        "csharp": (
            "GetBytes", "Fill", "GetInt32", "NextBytes", "Next",
            "NextInt64", "m05CreateBytes", "m05RandomSource",
        ),
        "python": (
            "urandom", "token_bytes", "token_hex", "token_urlsafe",
            "randbits", "randbelow", "randbytes",
            "randint", "randrange", "_m05_random_source",
            "_m05_create_bytes",
        ),
    }[analyzer.language]
    for call in analyzer.calls(*candidates):
        if not _is_random_source_call(analyzer, call):
            continue
        statement_start = max(
            analyzer.source.rfind(";", 0, call.start),
            analyzer.source.rfind("\n", 0, call.start),
        ) + 1
        statement_end = analyzer.source.find(";", call.end)
        if statement_end == -1:
            statement_end = min(len(analyzer.source), call.end + 200)
        context = (
            (analyzer.containing_function_name(call.start) or "")
            + " "
            + analyzer.source[statement_start : statement_end + 1]
        )
        if _F9_RANDOM_OUTPUT_ROLE.search(context) is None:
            continue
        expression = analyzer.source[call.start : call.end]
        resolved = _resolve_f9_random_origin(analyzer, expression, call.start)
        findings.append(
            _finding_from_resolved(
                analyzer,
                "M05",
                "security_random_output_role",
                call.callee,
                expression,
                call.start,
                call.end,
                resolved,
                _random_label,
            )
        )
    if findings:
        return _prioritize_f9_output_findings(_dedupe_findings(findings))
    return _legacy_f9_random_source_findings(analyzer)


def _prioritize_f9_output_findings(
    findings: list[DetectorFinding],
) -> list[DetectorFinding]:
    """Prefer the explicit numeric challenge over unrelated random tokens."""

    if not findings:
        return findings
    priorities = {
        id(finding): _f9_output_role_priority(finding.sink_name)
        for finding in findings
    }
    maximum = max(priorities.values())
    return [
        finding
        for finding in findings
        if priorities[id(finding)] == maximum
    ]


def _f9_output_role_priority(name: str) -> int:
    compact = re.sub(r"[^A-Za-z0-9]", "", name).lower()
    if any(
        marker in compact
        for marker in (
            "otp", "pin", "securitycode", "verificationcode",
            "accesscode", "authenticationcode", "authcode",
        )
    ):
        return 7
    if re.match(r"(?:generate|create|draw|next|run|issue|make)", compact) and any(
        marker in compact
        for marker in (
            "randomint", "randominteger", "randomnumber", "randomvalue",
            "challengeinteger", "challengenumber", "challengevalue",
            "otp", "pin", "securitycode", "verificationcode",
        )
    ):
        return 7
    if re.match(r"(?:generate|create|draw|next|run|issue|make)", compact) and any(
        marker in compact
        for marker in ("challenge", "random", "otp", "pin", "code", "nonce")
    ):
        return 6
    if any(
        marker in compact
        for marker in (
            "otp", "pin", "securitycode", "verificationcode",
            "accesscode", "authenticationcode", "authcode",
        )
    ):
        return 5
    if "nonce" in compact and any(
        marker in compact for marker in ("integer", "number", "value", "int", "code")
    ):
        return 5
    if (
        "challenge" in compact
        or "randomint" in compact
        or "randominteger" in compact
        or "randomnumber" in compact
        or "randomvalue" in compact
        or "operationvalue" in compact
        or compact in {
            "code",
            "codealt",
            "alternatecode",
            "alternativecode",
            "numericcode",
        }
    ):
        return 4
    if "nonce" in compact:
        return 3
    if "token" in compact and any(
        marker in compact for marker in ("integer", "number", "value", "code")
    ):
        return 2
    return 1


def _f9_integer_generator_findings(
    analyzer: SourceAnalyzer,
) -> list[DetectorFinding]:
    """Find the generated integer governed by F9_T3.

    Rewrites often pass the value directly to a wrapper, so there may be no
    assignment named ``challenge``.  Prefer called generator helpers and
    role-named parameters over downstream token/hash assignments; this keeps
    an unrelated CSPRNG key or a deterministic token derivation from becoming
    the controlled M05 decision.
    """

    if isinstance(analyzer, PythonAnalyzer):
        return _python_f9_integer_generator_findings(analyzer)

    findings: list[DetectorFinding] = []
    for definition in getattr(analyzer, "functions", ()):
        if (
            not _is_f9_integer_generator_name(definition.name)
            or not _f9_definition_returns_integer(analyzer, definition)
        ):
            continue
        calls = tuple(
            call
            for call in analyzer.calls(definition.name)
            if not definition.start <= call.start < definition.end
        )
        for call in calls:
            expression = analyzer.source[call.start : call.end]
            resolved = _resolve_f9_random_origin(analyzer, expression, call.start)
            if resolved.origin not in {"secure_random", "derived_secure", "non_csprng", "predictable"}:
                continue
            if (
                resolved.origin == "predictable"
                and not _resolved_value_has_random_source_evidence(analyzer, resolved)
            ):
                continue
            findings.append(
                _finding_from_resolved(
                    analyzer,
                    "M05",
                    "security_random_output_role",
                    definition.name,
                    expression,
                    call.start,
                    call.end,
                    resolved,
                    _random_label,
                )
            )
    if findings:
        return _prioritize_f9_output_findings(_dedupe_findings(findings))

    for sink_name, expression, start, end in _f9_role_parameter_arguments(analyzer):
        resolved = _resolve_f9_random_origin(analyzer, expression, start)
        if resolved.origin not in {"secure_random", "derived_secure", "non_csprng", "predictable"}:
            continue
        if (
            resolved.origin == "predictable"
            and not _resolved_value_has_random_source_evidence(analyzer, resolved)
        ):
            continue
        findings.append(
            _finding_from_resolved(
                analyzer,
                "M05",
                "security_random_output_role",
                sink_name,
                expression,
                start,
                end,
                resolved,
                _random_label,
            )
        )
    return _dedupe_findings(findings)


def _python_f9_integer_generator_findings(
    analyzer: PythonAnalyzer,
) -> list[DetectorFinding]:
    try:
        tree = ast.parse(analyzer.source)
    except SyntaxError:
        return []
    functions = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    terminal_findings: list[DetectorFinding] = []
    for call in analyzer.calls("print"):
        call_findings: list[DetectorFinding] = []
        for expression in call.arguments:
            resolved = _resolve_f9_random_origin(
                analyzer,
                expression,
                call.start,
            )
            if resolved.origin not in {
                "secure_random",
                "derived_secure",
                "non_csprng",
                "predictable",
            }:
                continue
            if (
                resolved.origin == "predictable"
                and not _resolved_value_has_random_source_evidence(
                    analyzer,
                    resolved,
                )
            ):
                continue
            call_findings.append(
                _finding_from_resolved(
                    analyzer,
                    "M05",
                    "security_random_output_role",
                    expression,
                    expression,
                    call.start,
                    call.end,
                    resolved,
                    _random_label,
                )
            )
        if len(call_findings) == 1:
            terminal_findings.extend(call_findings)
    if terminal_findings:
        return _prioritize_f9_output_findings(
            _dedupe_findings(terminal_findings)
        )

    findings: list[DetectorFinding] = []
    for definition in functions:
        if not _is_f9_integer_generator_name(definition.name):
            continue
        annotation = ast.unparse(definition.returns) if definition.returns else ""
        compact_name = re.sub(
            r"[^A-Za-z0-9]", "", definition.name
        ).lower()
        if annotation and annotation not in {"int", "builtins.int"}:
            continue
        numeric_name = bool(
            _F9_STRONG_INTEGER_OUTPUT_ROLE.fullmatch(definition.name)
            or _python_f9_name_is_numeric(definition.name)
        )
        if (
            not numeric_name
            and annotation not in {"int", "builtins.int"}
            and not (
                "challenge" in compact_name
                and _python_f9_definition_has_integer_return(definition)
            )
        ):
            continue
        for call in analyzer.calls(definition.name):
            expression = analyzer.source[call.start : call.end]
            resolved = _resolve_f9_random_origin(analyzer, expression, call.start)
            if resolved.origin not in {"secure_random", "derived_secure", "non_csprng", "predictable"}:
                continue
            if (
                resolved.origin == "predictable"
                and not _resolved_value_has_random_source_evidence(analyzer, resolved)
            ):
                continue
            findings.append(
                _finding_from_resolved(
                    analyzer,
                    "M05",
                    "security_random_output_role",
                    definition.name,
                    expression,
                    call.start,
                    call.end,
                    resolved,
                    _random_label,
                )
            )
    if findings:
        return _prioritize_f9_output_findings(_dedupe_findings(findings))

    for definition in functions:
        parameters = tuple(definition.args.posonlyargs) + tuple(definition.args.args) + tuple(
            definition.args.kwonlyargs
        )
        role_parameters = [
            parameter
            for parameter in parameters
            if _F9_RANDOM_OUTPUT_ROLE.fullmatch(parameter.arg) is not None
            and (
                _F9_STRONG_INTEGER_OUTPUT_ROLE.fullmatch(parameter.arg) is not None
                or (
                    parameter.annotation is not None
                    and ast.unparse(parameter.annotation) in {"int", "builtins.int"}
                )
            )
        ]
        if not role_parameters:
            continue
        parameter_names = [parameter.arg for parameter in parameters]
        for call in analyzer.calls(definition.name):
            for parameter in role_parameters:
                parameter_index = parameter_names.index(parameter.arg)
                expression = _call_argument(
                    call,
                    parameter_index,
                    parameter.arg,
                )
                if expression is None:
                    continue
                resolved = _resolve_f9_random_origin(analyzer, expression, call.start)
                if resolved.origin not in {"secure_random", "derived_secure", "non_csprng", "predictable"}:
                    continue
                if (
                    resolved.origin == "predictable"
                    and not _resolved_value_has_random_source_evidence(analyzer, resolved)
                ):
                    continue
                sink_name = (
                    definition.name
                    if _is_f9_integer_generator_name(definition.name)
                    else parameter.arg
                )
                findings.append(
                    _finding_from_resolved(
                        analyzer,
                        "M05",
                        "security_random_output_role",
                        sink_name,
                        expression,
                        call.start,
                        call.end,
                        resolved,
                        _random_label,
                    )
                )
    if findings:
        return _prioritize_f9_output_findings(_dedupe_findings(findings))

    for node in ast.walk(tree):
        target: ast.Name | None = None
        value: ast.expr | None = None
        annotated_integer = False
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0] if isinstance(node.targets[0], ast.Name) else None
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target
            value = node.value
            annotated_integer = ast.unparse(node.annotation) in {"int", "builtins.int"}
        if target is None or value is None:
            continue
        if isinstance(value, ast.Call):
            callee = (
                value.func.id
                if isinstance(value.func, ast.Name)
                else value.func.attr
                if isinstance(value.func, ast.Attribute)
                else None
            )
            matching = tuple(
                definition
                for definition in functions
                if callee is not None and definition.name == callee
            )
            annotations = tuple(
                ast.unparse(definition.returns)
                for definition in matching
                if definition.returns is not None
            )
            if annotations and all(
                annotation not in {"int", "builtins.int"}
                for annotation in annotations
            ):
                continue
        compact_target = re.sub(r"[^A-Za-z0-9]", "", target.id).lower()
        containing = next(
            (
                definition
                for definition in functions
                if definition.lineno <= node.lineno <= (definition.end_lineno or definition.lineno)
            ),
            None,
        )
        internal_integer_role = bool(
            containing is not None
            and target.id.lower() in {"value", "digit", "index", "number", "candidate"}
            and _is_f9_integer_generator_name(containing.name)
            and _python_f9_expression_is_integer_source(value)
        )
        if (
            "nonce" in compact_target
            and not annotated_integer
            and not _python_f9_expression_is_integer_source(value)
        ):
            continue
        if (
            _F9_STRONG_INTEGER_OUTPUT_ROLE.fullmatch(target.id) is None
            and not internal_integer_role
            and not (
                _F9_RANDOM_OUTPUT_ROLE.fullmatch(target.id) is not None
                and annotated_integer
            )
        ):
            continue
        start, end = _python_ast_span(analyzer.source, value)
        expression = analyzer.source[start:end]
        resolved = _resolve_f9_random_origin(analyzer, expression, start)
        if resolved.origin not in {"secure_random", "derived_secure", "non_csprng", "predictable"}:
            continue
        if (
            resolved.origin == "predictable"
            and not _resolved_value_has_random_source_evidence(analyzer, resolved)
        ):
            continue
        findings.append(
            _finding_from_resolved(
                analyzer,
                "M05",
                "security_random_output_role",
                target.id,
                expression,
                start,
                end,
                resolved,
                _random_label,
            )
        )
    if findings:
        return _prioritize_f9_output_findings(_dedupe_findings(findings))
    return _python_f9_structured_role_findings(
        analyzer,
        tree,
        functions,
    )


def _python_f9_structured_role_findings(
    analyzer: PythonAnalyzer,
    tree: ast.AST,
    functions: tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...],
) -> list[DetectorFinding]:
    """Resolve Python challenge values embedded in calls and payloads."""

    candidates: list[tuple[int, DetectorFinding]] = []

    def add(name: str, value: ast.expr, sink_name: str | None = None) -> None:
        # The value's own role controls eligibility. A helper such as
        # create_hmac_challenge may also receive a key or timestamp; its name
        # must not promote those unrelated arguments into M05 output sinks.
        priority = _f9_output_role_priority(name)
        if priority <= 1:
            return
        expressions: list[ast.expr] = [value]
        expressions.extend(
            child.value if isinstance(child, ast.NamedExpr) else child
            for child in ast.walk(value)
            if child is not value
            and isinstance(child, (ast.Call, ast.NamedExpr))
            and _python_f9_expression_is_integer_source(
                child.value if isinstance(child, ast.NamedExpr) else child
            )
        )
        seen_spans: set[tuple[int, int]] = set()
        for expression_index, candidate_expression in enumerate(expressions):
            start, end = _python_ast_span(analyzer.source, candidate_expression)
            if (start, end) in seen_spans:
                continue
            seen_spans.add((start, end))
            expression = analyzer.source[start:end]
            resolved = _resolve_f9_random_origin(analyzer, expression, start)
            if resolved.origin not in {
                "secure_random", "derived_secure", "non_csprng", "predictable",
            }:
                continue
            if (
                resolved.origin == "predictable"
                and not _resolved_value_has_random_source_evidence(analyzer, resolved)
            ):
                continue
            candidate_priority = priority + (
                10
                if _python_f9_expression_is_integer_source(candidate_expression)
                else 0
            )
            candidates.append(
                (
                    candidate_priority,
                    _finding_from_resolved(
                        analyzer,
                        "M05",
                        "security_random_output_role",
                        sink_name or name,
                        expression,
                        start,
                        end,
                        resolved,
                        _random_label,
                    ),
                )
            )
            if expression_index == 0:
                return

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = (
                node.targets[0]
                if isinstance(node, ast.Assign) and len(node.targets) == 1
                else node.target
                if isinstance(node, ast.AnnAssign)
                else None
            )
            value = node.value
            if isinstance(target, ast.Name) and value is not None:
                if not isinstance(value, (ast.Dict, ast.List, ast.Set, ast.Tuple)):
                    add(target.id, value)
                for joined in (
                    child for child in ast.walk(value)
                    if isinstance(child, ast.JoinedStr)
                ):
                    for index, item in enumerate(joined.values):
                        if not isinstance(item, ast.FormattedValue) or index == 0:
                            continue
                        previous = joined.values[index - 1]
                        prefix = (
                            previous.value
                            if isinstance(previous, ast.Constant)
                            and isinstance(previous.value, str)
                            else ""
                        )
                        if re.search(
                            r"(?i)(?:challenge|nonce|otp|pin|"
                            r"security[_ -]?code)\s*[=:]\s*$",
                            prefix,
                        ):
                            add("challenge_value", item.value, target.id)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and isinstance(value, ast.expr)
                ):
                    role = (
                        "challenge_value"
                        if key.value.lower() in {"ch", "chal"}
                        else key.value
                    )
                    add(role, value)
        elif isinstance(node, ast.Call):
            callee = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else ""
            )
            for keyword in node.keywords:
                if keyword.arg is not None:
                    role = keyword.arg
                    if (
                        role.lower() in {"value", "number", "code", "integer"}
                        and "challenge" in callee.lower()
                    ):
                        role = "challenge_value"
                    add(role, keyword.value, callee or None)

    definitions: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for definition in functions:
        definitions.setdefault(definition.name, []).append(definition)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else None
        )
        if callee is None:
            continue
        for definition in definitions.get(callee, ()):
            parameters = (
                tuple(definition.args.posonlyargs)
                + tuple(definition.args.args)
                + tuple(definition.args.kwonlyargs)
            )
            values: dict[str, ast.expr] = {
                parameter.arg: argument
                for parameter, argument in zip(parameters, node.args)
            }
            values.update(
                {
                    keyword.arg: keyword.value
                    for keyword in node.keywords
                    if keyword.arg is not None
                }
            )
            for parameter in parameters:
                value = values.get(parameter.arg)
                if value is not None:
                    add(parameter.arg, value, definition.name)

    if not candidates:
        return []
    maximum = max(priority for priority, _ in candidates)
    return _dedupe_findings(
        [finding for priority, finding in candidates if priority == maximum]
    )


def _python_f9_name_is_numeric(name: str) -> bool:
    compact = re.sub(r"[^A-Za-z0-9]", "", name).lower()
    return bool(
        any(marker in compact for marker in ("integer", "number", "code", "value"))
        or compact.endswith("int")
    )


def _python_f9_definition_has_integer_return(
    definition: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    assignments: dict[str, ast.expr] = {}
    for node in ast.walk(definition):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name):
                assignments[node.targets[0].id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                assignments[node.target.id] = node.value
    for node in ast.walk(definition):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        value = assignments.get(node.value.id) if isinstance(node.value, ast.Name) else node.value
        if value is not None and _python_f9_expression_is_integer_source(value):
            return True
    return False


def _python_f9_expression_is_integer_source(value: ast.expr) -> bool:
    text = ast.unparse(value)
    return bool(
        re.search(
            r"(?:\b(?:randbelow|randbits|randrange|randint|getrandbits)\s*\(|"
            r"\btime(?:_ns)?\s*\(|\btime_ns\s*\()",
            text,
        )
    )


def _python_ast_span(source: str, node: ast.AST) -> tuple[int, int]:
    lines = source.splitlines(keepends=True)
    starts: list[int] = []
    offset = 0
    for line in lines:
        starts.append(offset)
        offset += len(line)
    start = starts[node.lineno - 1] + node.col_offset
    end_line = getattr(node, "end_lineno", node.lineno)
    end_column = getattr(node, "end_col_offset", node.col_offset)
    end = starts[end_line - 1] + end_column
    return start, end


def _is_f9_integer_generator_name(name: str) -> bool:
    compact = re.sub(r"[^A-Za-z0-9]", "", name).lower()
    if not re.match(
        r"(?:generate|create|draw|next|issue|make|new|random|secure|crypto|pseudo)",
        compact,
    ):
        return False
    numeric = any(
        marker in compact
        for marker in ("integer", "number", "numeric", "code", "value", "pin", "otp")
    ) or compact.endswith("int")
    if any(marker in compact for marker in ("hash", "digest", "signature")):
        return False
    if "token" in compact and not numeric:
        return False
    return bool(
        numeric
        or
        "challenge" in compact
        or "random" in compact
        or "securitycode" in compact
        or "pin" in compact
        or "otp" in compact
        or "operationvalue" in compact
        or ("operation" in compact and numeric)
        or ("token" in compact and numeric)
    )


def _f9_definition_returns_integer(
    analyzer: SourceAnalyzer,
    definition: FunctionDefinition,
) -> bool:
    compact_name = re.sub(r"[^A-Za-z0-9]", "", definition.name).lower()
    header_start = definition.start
    if analyzer.language == "csharp":
        header_start = analyzer.source.rfind("\n", 0, definition.start) + 1
    header = analyzer.source[
        header_start : min(definition.end, definition.start + 500)
    ]
    escaped = re.escape(definition.name)
    if analyzer.language == "typescript":
        annotated_number = bool(
            re.search(
                rf"\b{escaped}\s*\([^)]*\)\s*:\s*"
                rf"(?:Promise\s*<\s*)?(?:number|bigint)\b",
                header,
            )
        )
        if annotated_number:
            return True
        body = analyzer.source[definition.start : definition.end]
        return bool(
            re.search(
                r"\breturn\s+(?!\s*[\{\[])[^;\n]*(?:\brandomInt\s*\(|"
                r"\bMath\s*\.\s*random\s*\(|\bDate\s*\.\s*now\s*\()",
                body,
            )
        )
    if analyzer.language == "csharp":
        return bool(
            re.search(
                rf"\b(?:s?byte|u?short|u?int|u?long|Int(?:16|32|64)|"
                rf"UInt(?:16|32|64))\s+{escaped}\s*\(",
                header,
            )
        )
    if analyzer.language == "python":
        return bool(
            re.search(
                rf"\bdef\s+{escaped}\s*\([^)]*\)\s*->\s*int\b",
                header,
            )
        )
    return False


def _f9_assignment_is_integer_role(
    analyzer: SourceAnalyzer,
    assignment: object,
) -> bool:
    name = getattr(assignment, "name")
    start = int(getattr(assignment, "start"))
    prefix = analyzer.source[max(0, start - 220) : start + len(name) + 80]
    escaped = re.escape(name)
    expression = str(getattr(assignment, "expression", ""))
    role_name = _F9_RANDOM_OUTPUT_ROLE.fullmatch(name) is not None
    function_name = (analyzer.containing_function_name(start) or "").lower()
    internal_challenge_digit = bool(
        name.lower() in {"value", "digit", "index", "number", "candidate"}
        and any(
            marker in function_name
            for marker in ("challenge", "otp", "pin", "code", "numeric")
        )
    )
    if not role_name and not internal_challenge_digit:
        return False
    if analyzer.language == "typescript":
        object_like_expression = re.match(
            r"\s*(?:\{|\[|new\s+(?!Uint8Array\b))",
            expression,
        ) is not None
        return bool(
            re.search(rf"\b{escaped}\s*:\s*(?:number|bigint)\b", prefix)
            or (
                not object_like_expression
                and re.search(
                    r"(?:\brandomInt\s*\(|\brandomBytes\s*\(|"
                    r"\bMath\s*\.\s*random\s*\(|"
                    r"\bDate\s*\.\s*now\s*\()",
                    expression,
                )
            )
        )
    if analyzer.language == "csharp":
        direct_random_integer = re.search(
            r"(?:\bRandomNumberGenerator\s*\.\s*GetInt32\s*\(|"
            r"\bnew\s+(?:System\.)?Random\s*\()",
            expression,
        )
        direct_clock_integer = (
            re.search(
                r"(?:\bDateTimeOffset\b|\bEnvironment\s*\.\s*TickCount\b)",
                expression,
            )
            is not None
            and re.match(r"\s*new\s+(?!\s*(?:System\.)?Random\b)", expression)
            is None
        )
        return bool(
            re.search(
                rf"\b(?:s?byte|u?short|u?int|u?long|Int(?:16|32|64)|"
                rf"UInt(?:16|32|64))\s+{escaped}\b",
                prefix,
            )
            or direct_random_integer
            or direct_clock_integer
        )
    if analyzer.language == "python":
        return bool(re.search(rf"\b{escaped}\s*:\s*int\b", prefix))
    return False


def _f9_assignment_is_byte_role(
    analyzer: SourceAnalyzer,
    assignment: object,
) -> bool:
    name = str(getattr(assignment, "name"))
    if _F9_RANDOM_OUTPUT_ROLE.fullmatch(name) is None:
        return False
    start = int(getattr(assignment, "start"))
    expression = str(getattr(assignment, "expression", ""))
    prefix = analyzer.source[max(0, start - 260) : start + len(name) + 120]
    escaped = re.escape(name)
    if analyzer.language == "typescript":
        return bool(
            re.search(rf"\b{escaped}\s*:\s*(?:Buffer|Uint8Array)\b", prefix)
            or re.fullmatch(
                r"\s*(?:randomBytes|drawRandomBytes|createChallengeBytes)"
                r"\s*\([^;]*\)\s*",
                expression,
                flags=re.DOTALL,
            )
        )
    if analyzer.language == "csharp":
        return bool(
            re.search(rf"\bbyte\s*\[\s*\]\s+{escaped}\b", prefix)
            or re.fullmatch(
                r"\s*(?:(?:System\s*\.\s*Security\s*\.\s*Cryptography\s*\.\s*)?"
                r"RandomNumberGenerator\s*\.\s*GetBytes|DrawRandomBytes|"
                r"CreateChallengeBytes|GenerateRandomBytes)\s*\([^;]*\)\s*",
                expression,
                flags=re.DOTALL,
            )
        )
    if analyzer.language == "python":
        return bool(
            re.search(rf"\b{escaped}\s*:\s*(?:bytes|bytearray)\b", prefix)
            or re.fullmatch(
                r"\s*(?:secrets\s*\.\s*token_bytes|os\s*\.\s*urandom|"
                r"random\s*\.\s*randbytes|_create_challenge_bytes)"
                r"\s*\([^;]*\)\s*",
                expression,
                flags=re.DOTALL,
            )
        )
    return False


def _f9_role_parameter_arguments(
    analyzer: SourceAnalyzer,
) -> list[tuple[str, str, int, int]]:
    values: list[tuple[str, str, int, int]] = []
    for definition in getattr(analyzer, "functions", ()):
        role_indexes = [
            index
            for index, parameter in enumerate(definition.parameters)
            if (
                _F9_RANDOM_OUTPUT_ROLE.fullmatch(parameter) is not None
                or _f9_contextual_parameter_role(definition.name, parameter)
            )
            and _f9_parameter_is_integer(analyzer, definition, parameter)
        ]
        if not role_indexes:
            continue
        for call in analyzer.calls(definition.name):
            if definition.start <= call.start < definition.end:
                continue
            for parameter_index in role_indexes:
                expression = _call_argument(
                    call,
                    parameter_index,
                    definition.parameters[parameter_index],
                )
                if expression is None:
                    continue
                values.append(
                    (definition.name, expression, call.start, call.end)
                )

    if isinstance(analyzer, CStyleAnalyzer) and analyzer.language == "typescript":
        for match in re.finditer(
            r"\(\s*(?:async\s+)?\(\s*(?P<parameters>[^()]*)\)\s*=>\s*\{",
            analyzer.masked,
        ):
            body_open = match.end() - 1
            body_close = _matching_c_style_brace(analyzer.masked, body_open)
            if body_close == -1:
                continue
            cursor = body_close + 1
            while cursor < len(analyzer.masked) and analyzer.masked[cursor].isspace():
                cursor += 1
            if cursor >= len(analyzer.masked) or analyzer.masked[cursor] != ")":
                continue
            cursor += 1
            while cursor < len(analyzer.masked) and analyzer.masked[cursor].isspace():
                cursor += 1
            if cursor >= len(analyzer.masked) or analyzer.masked[cursor] != "(":
                continue
            call_close = _matching_c_style_parenthesis(analyzer.masked, cursor)
            if call_close == -1:
                continue
            raw_parameters = split_arguments(
                f"({match.group('parameters')})",
                0,
                len(match.group("parameters")) + 1,
            )
            parameters = tuple(
                (
                    re.match(r"[A-Za-z_$][A-Za-z0-9_$]*", item.strip()).group(0),
                    item,
                )
                for item in raw_parameters
                if re.match(r"[A-Za-z_$][A-Za-z0-9_$]*", item.strip())
            )
            arguments = split_arguments(analyzer.source, cursor, call_close)
            for parameter_index, (parameter, raw_parameter) in enumerate(parameters):
                if (
                    _F9_RANDOM_OUTPUT_ROLE.fullmatch(parameter) is None
                    or (
                        _F9_STRONG_INTEGER_OUTPUT_ROLE.fullmatch(parameter) is None
                        and re.search(
                            rf"\b{re.escape(parameter)}\s*:\s*"
                            rf"(?:number|bigint)\b",
                            raw_parameter,
                        )
                        is None
                    )
                    or parameter_index >= len(arguments)
                ):
                    continue
                values.append(
                    (
                        parameter,
                        _argument_value(arguments[parameter_index]),
                        cursor,
                        call_close + 1,
                    )
                )
    return values


def _f9_contextual_parameter_role(function_name: str, parameter: str) -> bool:
    parameter_compact = re.sub(r"[^A-Za-z0-9]", "", parameter).lower()
    function_compact = re.sub(r"[^A-Za-z0-9]", "", function_name).lower()
    return bool(
        parameter_compact in {
            "value", "digit", "index", "number", "candidate", "numericpart",
        }
        and any(
            marker in function_compact
            for marker in ("challenge", "otp", "pin", "code", "numeric")
        )
    )


def _f9_parameter_is_integer(
    analyzer: SourceAnalyzer,
    definition: FunctionDefinition,
    parameter: str,
) -> bool:
    header = analyzer.source[
        definition.start : min(definition.end, definition.start + 700)
    ]
    escaped = re.escape(parameter)
    if analyzer.language == "typescript":
        return bool(re.search(rf"\b{escaped}\s*:\s*(?:number|bigint)\b", header))
    if analyzer.language == "csharp":
        return bool(
            re.search(
                rf"\b(?:s?byte|u?short|u?int|u?long|Int(?:16|32|64)|"
                rf"UInt(?:16|32|64))\s+{escaped}\b",
                header,
            )
        )
    if analyzer.language == "python":
        return bool(re.search(rf"\b{escaped}\s*:\s*int\b", header))
    return False




def _csharp_f9_direct_random_api_origin(
    analyzer: SourceAnalyzer,
    expression: str,
    start: int,
) -> ValueOrigin | None:
    """Classify native C# random calls inside the controlled F9 expression."""

    if analyzer.language != "csharp":
        return None
    leading_space = len(expression) - len(expression.lstrip())
    expression_start = start + leading_space
    expression_end = expression_start + len(expression.strip())
    origins: set[ValueOrigin] = set()
    for call in analyzer.calls(
        "GetBytes",
        "Fill",
        "GetInt32",
        "NextBytes",
        "Next",
        "NextInt64",
    ):
        if call.start < expression_start or call.end > expression_end:
            continue
        if not _is_random_source_call(analyzer, call):
            continue
        origin = _random_source_api_origin(analyzer, call)
        if origin is not None:
            origins.add(origin)
    if "non_csprng" in origins:
        return "non_csprng"
    if "secure_random" in origins:
        return "secure_random"
    return None


def _resolve_f9_random_origin(
    analyzer: SourceAnalyzer,
    expression: str,
    start: int,
) -> ResolvedValue:
    stripped = expression.strip()
    walrus_text = (
        stripped[1:-1].strip()
        if stripped.startswith("(") and stripped.endswith(")")
        else stripped
    )
    walrus = re.fullmatch(
        r"[A-Za-z_]\w*\s*:=\s*(?P<value>.+)",
        walrus_text,
        flags=re.DOTALL,
    )
    parser_tail = re.fullmatch(r"=\s*(?P<value>.+)", stripped, flags=re.DOTALL)
    wrapped_value = walrus or parser_tail
    if wrapped_value is not None:
        value = wrapped_value.group("value").strip()
        value_start = start + expression.find(value)
        return _resolve_f9_random_origin(analyzer, value, value_start)

    resolved = analyzer.resolve_origin(expression, start)
    if resolved.origin == "unknown" and any(
        step.action == "reject native RNG callable used as generator"
        for step in resolved.steps
    ):
        # A proven nonexistent method cannot acquire entropy merely because
        # its wrapper stores a native RNG function as a constructor argument.
        return resolved
    for candidate_expression, candidate_index in (
        (resolved.expression, start),
        *(
            (
                step.expression,
                step.span.start if step.span.start >= 0 else start,
            )
            for step in resolved.steps
        ),
    ):
        mutated_origin = _f9_mutated_buffer_origin(
            analyzer,
            candidate_expression,
            candidate_index,
        )
        if mutated_origin is not None:
            return replace(
                resolved, origin=mutated_origin, values=(),
                steps=resolved.steps + (ResolutionStep(
                    expression=candidate_expression,
                    action="resolve filled random buffer",
                    result=mutated_origin,
                    span=analyzer.source_span(candidate_index, candidate_index),
                ),),
            )
        if isinstance(analyzer, PythonAnalyzer):
            iterator_origin = _python_f9_bounded_iterator_origin(
                analyzer, candidate_expression, candidate_index,
            )
            if iterator_origin is not None:
                return replace(
                    resolved, origin=iterator_origin, values=(),
                    steps=resolved.steps + (ResolutionStep(
                        expression=candidate_expression,
                        action="resolve finite iterator before exhaustion",
                        result=iterator_origin,
                        span=analyzer.source_span(candidate_index, candidate_index),
                    ),),
                )
    if resolved.origin == "constant" and resolved.values:
        # Quoted labels such as ``"Random.randrange() code:"`` describe an
        # API but are not calls to it.  Do not feed literal text back through
        # the lexical random-source fallback below.
        return resolved
    if (
        resolved.origin in {"predictable", "non_csprng"}
        and any(
            step.action == "resolve template interpolation origins"
            for step in resolved.steps
        )
        and any(step.result == "secure_random" for step in resolved.steps)
    ):
        # Public timestamps and labels do not remove entropy from a separately
        # generated random component when the values are concatenated.
        return replace(resolved, origin="secure_random")
    if resolved.origin not in {"unknown", "constant"}:
        return resolved
    if resolved.origin == "unknown" and any(
        "conditional" in step.action
        and step.result == "unknown"
        for step in resolved.steps
    ):
        return resolved
    direct_api_origin = _csharp_f9_direct_random_api_origin(
        analyzer,
        expression,
        start,
    )
    if direct_api_origin is not None:
        return replace(resolved, origin=direct_api_origin)
    static_origins = _python_f9_resolved_random_origins(
        analyzer,
        resolved,
        start,
    )
    if static_origins:
        steps = resolved.steps + tuple(
            ResolutionStep(
                expression=expression,
                action="resolve selected callable alternative",
                result=origin,
                span=analyzer.source_span(start, start + len(expression)),
            )
            for origin in sorted(static_origins)
        )
        secure = static_origins & {"secure_random", "derived_secure"}
        weak = static_origins & {"non_csprng", "predictable"}
        if secure and weak:
            return replace(resolved, origin="unknown", steps=steps)
        if secure:
            return replace(resolved, origin="secure_random", steps=steps)
        if "non_csprng" in weak:
            return replace(resolved, origin="non_csprng", steps=steps)
        return replace(resolved, origin="predictable", steps=steps)
    if isinstance(analyzer, PythonAnalyzer):
        # Python lexical names are not evidence after bounded source resolution:
        # `random` may be a CSPRNG instance or an unrelated custom callback.
        return resolved
    evidence = "\n".join(
        (resolved.expression,) + tuple(step.expression for step in resolved.steps)
    )
    evidence += "\n" + _f9_explicit_source_evidence(
        analyzer,
        expression,
        start,
    )
    helper = re.fullmatch(
        r"(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*\([^)]*\)",
        expression.strip(),
    )
    if helper is not None:
        bodies = [
            analyzer.source[definition.start : definition.end]
            for definition in getattr(analyzer, "functions", ())
            if definition.name == helper.group("name")
        ]
        if bodies:
            evidence += "\n" + "\n".join(bodies)
    if re.search(
        r"(?:\brandomBytes\s*\(|\b(?:secureRandomInt|randomInt)\s*\(|"
        r"\brandomFill(?:Sync)?\s*\(|"
        r"\bRandomNumberGenerator\s*\.\s*(?:GetBytes|Fill|GetInt32)\s*\(|"
        r"\b(?:secrets\s*\.\s*(?:token_bytes|token_hex|token_urlsafe|"
        r"choice|randbits|randbelow)|"
        r"os\s*\.\s*urandom)\s*\()",
        evidence,
        flags=re.IGNORECASE,
    ):
        return replace(resolved, origin="secure_random")
    if re.search(
        r"(?:\bMath\s*\.\s*(?:random|imul)\s*\(|"
        r"\bnew\s+(?:System\.)?Random\s*\(|"
        r"\b(?:System\s*\.\s*)?Random\s*\.\s*Shared\s*\.\s*"
        r"(?:Next|NextBytes|NextInt64)\s*\(|"
        r"\brandom\s*\.\s*(?:random|randbytes|randint|randrange|getrandbits)\s*\()",
        evidence,
        flags=re.IGNORECASE,
    ):
        return replace(resolved, origin="non_csprng")
    csharp_receiver = re.search(
        r"\b(?P<name>[A-Za-z_]\w*)\s*\.\s*(?:Next|NextBytes|NextInt64)\s*\(",
        evidence,
    )
    if csharp_receiver is not None and re.search(
        rf"\b(?:System\.)?Random\s+{re.escape(csharp_receiver.group('name'))}\b",
        analyzer.source[:start],
    ):
        return replace(resolved, origin="non_csprng")
    if re.search(
        r"(?:\bDate\s*\.\s*now\s*\(|\bDateTimeOffset\b|"
        r"\bEnvironment\s*\.\s*TickCount\b|\btime\s*\.\s*time(?:_ns)?\s*\()",
        evidence,
        flags=re.IGNORECASE,
    ):
        return replace(resolved, origin="predictable")
    return resolved


def _python_f9_resolved_random_origins(
    analyzer: SourceAnalyzer,
    resolved: ResolvedValue,
    index: int,
) -> set[ValueOrigin]:
    if not isinstance(analyzer, PythonAnalyzer):
        return set()
    origins = _python_f9_static_random_origins(
        analyzer,
        resolved.expression,
        index,
    )
    for step in resolved.steps:
        step_index = step.span.start if step.span.start >= 0 else index
        origins.update(
            _python_f9_static_random_origins(
                analyzer,
                step.expression,
                step_index,
            )
        )
    return origins


def _python_f9_static_random_origins(
    analyzer: SourceAnalyzer,
    expression: str,
    index: int,
) -> set[ValueOrigin]:
    """Trace Python random sources through bounded deterministic wrappers."""

    if not isinstance(analyzer, PythonAnalyzer):
        return set()
    try:
        node = ast.parse(expression.strip(), mode="eval").body
    except SyntaxError:
        return set()
    return _python_f9_node_random_origins(
        analyzer,
        node,
        index,
        bindings={},
        seen=frozenset(),
    )


def _python_f9_node_random_origins(
    analyzer: PythonAnalyzer,
    node: ast.AST,
    index: int,
    *,
    bindings: dict[str, tuple[ast.AST, int]],
    seen: frozenset[tuple[int, int]],
) -> set[ValueOrigin]:
    key = (id(node), index)
    if key in seen:
        return set()
    next_seen = seen | frozenset({key})

    if isinstance(node, (ast.Name, ast.Attribute)):
        name = ast.unparse(node)
        direct = _python_f9_callable_origin(analyzer, name, index)
        if direct is not None:
            return {direct}
        if isinstance(node, ast.Name) and node.id in bindings:
            bound, bound_index = bindings[node.id]
            return _python_f9_node_random_origins(
                analyzer,
                bound,
                bound_index,
                bindings=bindings,
                seen=next_seen,
            )
        assignments = analyzer._reaching_assignments(name, index)
        origins: set[ValueOrigin] = set()
        for value, value_index in assignments:
            origins.update(
                _python_f9_node_random_origins(
                    analyzer,
                    value,
                    value_index,
                    bindings=bindings,
                    seen=next_seen,
                )
            )
        return origins

    if isinstance(node, ast.Call):
        callee = ast.unparse(node.func)
        direct = _python_f9_callable_origin(analyzer, callee, index)
        if analyzer.native_random_call_name(callee, index) is not None:
            return {direct} if direct is not None else set()
        origins = {direct} if direct is not None else set()

        if isinstance(node.func, ast.Name):
            if node.func.id in bindings:
                bound, _bound_index = bindings[node.func.id]
                if isinstance(bound, (ast.Name, ast.Attribute)):
                    origins.update(
                        _python_f9_function_random_origins(
                            analyzer,
                            ast.unparse(bound).rsplit(".", 1)[-1],
                            node,
                            index,
                            bindings=bindings,
                            seen=next_seen,
                        )
                    )
                    callable_origin = _python_f9_callable_origin(
                        analyzer,
                        ast.unparse(bound),
                        _bound_index,
                    )
                    if callable_origin is not None:
                        origins.add(callable_origin)
            else:
                aliased = _python_f9_plain_function_alias(analyzer, node.func.id)
                origins.update(
                    _python_f9_function_random_origins(
                        analyzer,
                        aliased or node.func.id,
                        node,
                        index,
                        bindings=bindings,
                        seen=next_seen,
                    )
                )

        if isinstance(node.func, ast.Attribute) and node.func.attr in {
            "digest",
            "hexdigest",
            "finalize",
            "read",
        }:
            receiver = ast.unparse(node.func.value)
            compact_receiver = re.sub(r"\s+", "", receiver)
            for update in analyzer.calls("update"):
                compact = re.sub(r"\s+", "", update.callee)
                if compact != f"{compact_receiver}.update":
                    continue
                if update.start >= index:
                    continue
                for argument in update.arguments:
                    origins.update(
                        _python_f9_expression_random_origins(
                            analyzer,
                            argument,
                            update.start,
                            bindings=bindings,
                            seen=next_seen,
                        )
                    )

        for child in (*node.args, *(item.value for item in node.keywords)):
            origins.update(
                _python_f9_node_random_origins(
                    analyzer,
                    child,
                    index,
                    bindings=bindings,
                    seen=next_seen,
                )
            )
        if isinstance(node.func, ast.Attribute):
            origins.update(
                _python_f9_node_random_origins(
                    analyzer,
                    node.func.value,
                    index,
                    bindings=bindings,
                    seen=next_seen,
                )
            )
        return origins

    origins: set[ValueOrigin] = set()
    for child in ast.iter_child_nodes(node):
        origins.update(
            _python_f9_node_random_origins(
                analyzer,
                child,
                index,
                bindings=bindings,
                seen=next_seen,
            )
        )
    return origins


def _python_f9_expression_random_origins(
    analyzer: PythonAnalyzer,
    expression: str,
    index: int,
    *,
    bindings: dict[str, tuple[ast.AST, int]],
    seen: frozenset[tuple[int, int]],
) -> set[ValueOrigin]:
    try:
        node = ast.parse(expression.strip(), mode="eval").body
    except SyntaxError:
        return set()
    return _python_f9_node_random_origins(
        analyzer,
        node,
        index,
        bindings=bindings,
        seen=seen,
    )


def _python_f9_function_random_origins(
    analyzer: PythonAnalyzer,
    name: str,
    call: ast.Call,
    index: int,
    *,
    bindings: dict[str, tuple[ast.AST, int]],
    seen: frozenset[tuple[int, int]],
) -> set[ValueOrigin]:
    origins: set[ValueOrigin] = set()
    for definition in analyzer._functions_by_name.get(name, ()):
        parameters = list(definition.args.posonlyargs) + list(
            definition.args.args
        )
        if parameters and parameters[0].arg in {"self", "cls"}:
            parameters = parameters[1:]
        local_bindings = dict(bindings)
        for parameter, argument in zip(parameters, call.args):
            local_bindings[parameter.arg] = (argument, index)
        local_bindings.update(
            {
                keyword.arg: (keyword.value, index)
                for keyword in call.keywords
                if keyword.arg is not None
            }
        )
        for returned in ast.walk(definition):
            if not isinstance(returned, ast.Return) or returned.value is None:
                continue
            return_index = analyzer._node_span(returned.value)[0]
            origins.update(
                _python_f9_node_random_origins(
                    analyzer,
                    returned.value,
                    return_index,
                    bindings=local_bindings,
                    seen=seen,
                )
            )
    return origins


def _python_f9_callable_origin(
    analyzer: PythonAnalyzer,
    name: str,
    index: int,
) -> ValueOrigin | None:
    native = analyzer.native_random_call_origin(name, index)
    if native is not None:
        return native
    compact = re.sub(r"\s+", "", name)
    head, separator, tail = compact.partition(".")
    canonical = analyzer.import_aliases.get(head, head) + (
        separator + tail if separator else ""
    )
    if canonical in {
        "time.time",
        "time.time_ns",
        "datetime.datetime.now",
        "datetime.datetime.utcnow",
    }:
        return "predictable"
    return None


def _python_f9_plain_function_alias(analyzer: PythonAnalyzer, name: str) -> str | None:
    """Follow one immutable module binding to a plain source-defined function."""
    assignments = analyzer._assignments.get(name, ())
    if len(assignments) != 1:
        return None
    value, _ = assignments[0]
    if any(isinstance(node, ast.arg) and node.arg == name for node in ast.walk(analyzer.tree)):
        return None
    if any(isinstance(node, (ast.Import, ast.ImportFrom)) and any(
        alias.name == "*" or (alias.asname or alias.name.split(".")[0]) == name
        for alias in node.names
    ) for node in ast.walk(analyzer.tree)):
        return None
    if not isinstance(value, ast.Name) or analyzer._enclosing_function(value) is not None:
        return None
    if analyzer._parents.get(analyzer._parents.get(value)) is not analyzer.tree:
        return None
    definitions = analyzer._functions_by_name.get(value.id, ())
    if len(definitions) != 1 or analyzer._assignments.get(value.id):
        return None
    function = definitions[0]
    if (
        not isinstance(function, ast.FunctionDef) or function.decorator_list
        or analyzer._parents.get(function) is not analyzer.tree
        or analyzer._node_span(function)[0] >= analyzer._node_span(value)[0]
        or any(owner.name == value.id for owner in analyzer._class_nodes)
        or any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(function))
    ):
        return None
    return value.id


def _python_f9_plain_field(
    analyzer: PythonAnalyzer, node: ast.AST, index: int,
) -> tuple[ast.ClassDef, ast.AST] | None:
    """A field initialized once by a plain constructor, with no patched dispatch."""
    if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == "self"):
        return None
    function = analyzer._containing_function(index)
    owner = analyzer._enclosing_class(function) if function is not None else None
    if owner is None or owner.bases or owner.decorator_list or owner.keywords:
        return None
    if analyzer._assignments.get(owner.name) or any(
        any(isinstance(base, ast.Name) and base.id == owner.name for base in other.bases)
        for other in analyzer._class_nodes
    ):
        return None
    if any(isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
        item.decorator_list or item.name in {"__getattribute__", "__getattr__", "__setattr__"}
    ) for item in owner.body):
        return None
    assignments = [
        value for value, _ in analyzer._assignments.get(ast.unparse(node), ())
        if analyzer._enclosing_class(value) is owner
    ]
    if len(assignments) != 1:
        return None
    value = assignments[0]
    constructor = analyzer._enclosing_function(value)
    if constructor is None or constructor.name != "__init__":
        return None
    statement = analyzer._parents.get(value)
    if analyzer._parents.get(statement) is not constructor:
        return None
    # Outside writes and dynamic mutation would invalidate the field identity.
    if any(name.endswith(f".{node.attr}") and name != ast.unparse(node)
           for name in analyzer._assignments):
        return None
    if any(
        isinstance(reference, ast.Attribute) and reference.attr == node.attr
        and analyzer._enclosing_class(reference) is not owner
        and not (isinstance(reference.value, ast.Name) and reference.value.id == "self"
                 and analyzer._enclosing_class(reference) is not None)
        for reference in ast.walk(analyzer.tree)
    ):
        # External reads can consume an iterator; augmented writes and deletes
        # are not ordinary assignments in the analyzer's binding index.
        return None
    if any(isinstance(call.func, ast.Name) and call.func.id in {"setattr", "delattr"}
           for call in analyzer._call_nodes):
        return None
    return owner, value


def _python_f9_call_count(
    analyzer: PythonAnalyzer, call: ast.Call, seen: frozenset[int] = frozenset(),
) -> int | None:
    """Upper-bound direct executions; unknown loops, callbacks and recursion stay unknown."""
    count = 1
    current: ast.AST = call
    parent = analyzer._parents.get(current)
    while parent is not None and not isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if isinstance(parent, (ast.While, ast.AsyncFor, ast.Lambda, ast.comprehension,
                               ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return None
        if isinstance(parent, ast.For):
            size = _python_f9_range_size(parent.iter)
            if size is None:
                return None
            count *= size
        current, parent = parent, analyzer._parents.get(parent)
    if parent is None:
        return count
    if id(parent) in seen or not isinstance(parent, ast.FunctionDef) or parent.decorator_list:
        return None
    if any(analyzer._enclosing_function(item) is parent and parent in analyzer._matching_functions(item)
           for item in analyzer._call_nodes):
        return None
    # Referring to the callable as a value can introduce invocations not counted here.
    for reference in ast.walk(analyzer.tree):
        if ((isinstance(reference, ast.Name) and reference.id == parent.name)
                or (isinstance(reference, ast.Attribute) and reference.attr == parent.name)):
            enclosing = analyzer._parents.get(reference)
            if not isinstance(enclosing, ast.Call) or enclosing.func is not reference:
                return None
    callers = [item for item in analyzer._call_nodes
               if analyzer._external_call_targets_function(item, parent)]
    if not callers:
        return None
    totals = [_python_f9_call_count(analyzer, item, seen | {id(parent)}) for item in callers]
    if any(total is None for total in totals):
        return None
    return count * sum(total for total in totals if total is not None)


def _python_f9_range_size(node: ast.AST) -> int | None:
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "range" and not node.keywords and 1 <= len(node.args) <= 3
            and all(isinstance(arg, ast.Constant) and type(arg.value) is int for arg in node.args)):
        return None
    try:
        return len(range(*(arg.value for arg in node.args)))
    except (ValueError, OverflowError):
        return None


def _python_f9_unshadowed_builtin(analyzer: PythonAnalyzer, name: str) -> bool:
    for node in ast.walk(analyzer.tree):
        if isinstance(node, ast.arg) and node.arg == name:
            return False
        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, (ast.Store, ast.Del)):
            return False
        if isinstance(node, ast.Attribute) and node.attr == name and isinstance(node.ctx, (ast.Store, ast.Del)):
            return False
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            return False
        if isinstance(node, (ast.Import, ast.ImportFrom)) and any(
            alias.name == "*" or (alias.asname or alias.name.split(".")[0]) == name
            for alias in node.names
        ):
            return False
    return True


def _python_f9_bounded_iterator_origin(
    analyzer: PythonAnalyzer, expression: str, index: int,
) -> ValueOrigin | None:
    """Resolve next(field, default) only when every next stays inside finite capacity."""
    try:
        node = ast.parse(expression.strip(), mode="eval").body
    except SyntaxError:
        return None
    if isinstance(node, ast.Subscript):
        node = node.value
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "next" and 1 <= len(node.args) <= 2 and not node.keywords):
        return None
    if not all(_python_f9_unshadowed_builtin(analyzer, name) for name in ("next", "range")):
        return None
    field = _python_f9_plain_field(analyzer, node.args[0], index)
    if field is None:
        return None
    owner, generator = field
    if not (isinstance(generator, ast.GeneratorExp) and len(generator.generators) == 1):
        return None
    loop = generator.generators[0]
    capacity = _python_f9_range_size(loop.iter)
    if capacity is None or capacity <= 0 or loop.ifs or loop.is_async:
        return None
    total = 0
    field_name = ast.unparse(node.args[0])
    for reference in ast.walk(owner):
        if not (isinstance(reference, ast.Attribute) and ast.unparse(reference) == field_name):
            continue
        if isinstance(reference.ctx, ast.Store):
            continue
        use = analyzer._parents.get(reference)
        if not (isinstance(use, ast.Call) and isinstance(use.func, ast.Name)
                and use.func.id == "next" and use.args and use.args[0] is reference):
            return None
        count = _python_f9_call_count(analyzer, use)
        if count is None:
            return None
        total += count
    if not 0 < total <= capacity:
        return None
    origin = analyzer.resolve_origin(ast.unparse(generator.elt), analyzer._node_span(generator.elt)[0]).origin
    return origin if origin in {"secure_random", "non_csprng"} else None


def _python_f9_filled_buffer_origin(
    analyzer: PythonAnalyzer, expression: str, index: int,
) -> ValueOrigin | None:
    """Recognize an initially empty buffer filled to a positive requested length."""
    try:
        node = ast.parse(expression.strip(), mode="eval").body
    except SyntaxError:
        return None
    if not (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice)
            and node.slice.lower is None and node.slice.step is None and node.slice.upper is not None):
        return None
    field = _python_f9_plain_field(analyzer, node.value, index)
    if field is None or not isinstance(field[1], ast.Constant) or field[1].value != b"":
        return None
    owner, _ = field
    function = analyzer._containing_function(index)
    if function is None or len(function.body) != 2:
        return None
    loop, returned = function.body
    if not (isinstance(loop, ast.While) and not loop.orelse and len(loop.body) == 1
            and isinstance(returned, ast.Return) and returned.value is not None
            and ast.unparse(returned.value) == ast.unparse(node)):
        return None
    condition = loop.test
    field_name = ast.unparse(node.value)
    if not (isinstance(condition, ast.Compare) and len(condition.ops) == 1
            and isinstance(condition.ops[0], ast.Lt) and len(condition.comparators) == 1
            and ast.unparse(condition.comparators[0]) == ast.unparse(node.slice.upper)
            and isinstance(condition.left, ast.Call) and isinstance(condition.left.func, ast.Name)
            and condition.left.func.id == "len" and len(condition.left.args) == 1
            and ast.unparse(condition.left.args[0]) == field_name):
        return None
    mutation = loop.body[0]
    if not (isinstance(mutation, ast.AugAssign) and isinstance(mutation.op, ast.Add)
            and ast.unparse(mutation.target) == field_name and isinstance(mutation.value, ast.Call)):
        return None
    mutations = [item for item in ast.walk(owner) if isinstance(item, ast.AugAssign)
                 and ast.unparse(item.target) == field_name]
    if mutations != [mutation] or not _python_f9_unshadowed_builtin(analyzer, "len"):
        return None
    size = analyzer.resolve_ints(ast.unparse(node.slice.upper), index)
    if not size.values or not all(re.fullmatch(r"[0-9]+", value) and int(value) > 0 for value in size.values):
        return None
    draw = mutation.value
    native = analyzer.native_random_call_name(ast.unparse(draw.func), analyzer._node_span(draw)[0])
    if native not in {"os.urandom", "secrets.token_bytes", "random.randbytes", "random.Random.randbytes", "random.SystemRandom.randbytes"}:
        return None
    if len(draw.args) != 1 or not isinstance(draw.args[0], ast.Constant) or type(draw.args[0].value) is not int or draw.args[0].value <= 0:
        return None
    return analyzer.native_random_call_origin(ast.unparse(draw.func), analyzer._node_span(draw)[0])


def _f9_explicit_source_evidence(
    analyzer: SourceAnalyzer,
    expression: str,
    index: int,
    seen: frozenset[tuple[str, int]] = frozenset(),
) -> str:
    """Follow the task value to the explicit PRNG choice used by wrappers."""

    normalized = expression.strip()
    key = (normalized, index)
    if key in seen:
        return normalized
    next_seen = seen | frozenset({key})
    evidence = normalized
    if not isinstance(analyzer, CStyleAnalyzer):
        return evidence

    reference = re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", normalized)
    if reference is not None:
        name = reference.group(0)
        escaped = re.escape(name)
        mutation_patterns = {
            "typescript": (
                rf"\b(?:randomFillSync|cryptoRandomFillSync)\s*\(\s*{escaped}\b|"
                rf"\b(?:crypto\s*\.\s*)?getRandomValues\s*\(\s*{escaped}\b"
            ),
            "csharp": (
                rf"\bRandomNumberGenerator\s*\.\s*Fill\s*\(\s*{escaped}\b|"
                rf"\b(?:System\s*\.\s*)?Random\s*\.\s*Shared\s*\.\s*"
                rf"NextBytes\s*\(\s*{escaped}\b|"
                rf"\b(?:new\s+(?:System\.)?Random\s*\([^;\n]*\)|"
                rf"[A-Za-z_]\w*)\s*\.\s*NextBytes\s*\(\s*{escaped}\b"
            ),
        }
        mutation_pattern = mutation_patterns.get(analyzer.language)
        if mutation_pattern is not None:
            mutations = tuple(
                match
                for match in re.finditer(
                    mutation_pattern,
                    analyzer.source[:index],
                )
                if analyzer.position_is_in_same_or_enclosing_function(
                    match.start(),
                    index,
                )
            )
            if mutations:
                evidence += "\n" + mutations[-1].group(0)
        assignment = analyzer.latest_assignment(name, index)
        if assignment is not None:
            evidence += "\n" + _f9_explicit_source_evidence(
                analyzer,
                assignment.expression,
                assignment.start,
                next_seen,
            )
        containing = min(
            (
                definition
                for definition in analyzer.functions
                if definition.start <= index <= definition.end
                and name in definition.parameters
            ),
            key=lambda definition: definition.end - definition.start,
            default=None,
        )
        if containing is not None:
            parameter_index = containing.parameters.index(name)
            for call in analyzer.calls(containing.name):
                if containing.start <= call.start <= containing.end:
                    continue
                argument = _call_argument(
                    call,
                    parameter_index,
                    name,
                )
                if argument is None:
                    continue
                evidence += "\n" + _f9_explicit_source_evidence(
                    analyzer,
                    argument,
                    call.start,
                    next_seen,
                )

    helper = re.match(r"\s*(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*\(", normalized)
    if helper is not None:
        for definition in analyzer.functions:
            if definition.name != helper.group("name"):
                continue
            for returned in definition.returns:
                evidence += "\n" + _f9_explicit_source_evidence(
                    analyzer,
                    returned.expression,
                    returned.index,
                    next_seen,
                )
    return evidence


def _f9_mutated_buffer_origin(
    analyzer: SourceAnalyzer,
    expression: str,
    start: int,
) -> ValueOrigin | None:
    """Recover the random origin of a buffer filled before its sink use."""

    if isinstance(analyzer, PythonAnalyzer):
        return _python_f9_filled_buffer_origin(analyzer, expression, start)
    if not isinstance(analyzer, CStyleAnalyzer):
        return None
    variable = re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", expression.strip())
    if variable is None:
        return None
    name = re.escape(variable.group(0))
    prefix = analyzer.source[:start]
    candidates: list[tuple[int, ValueOrigin]] = []

    patterns: tuple[tuple[str, ValueOrigin], ...] = (
        (
            rf"\b(?:randomFillSync|cryptoRandomFillSync)\s*\(\s*{name}\b",
            "secure_random",
        ),
        (
            rf"\b(?:crypto\s*\.\s*)?getRandomValues\s*\(\s*{name}\b",
            "secure_random",
        ),
        (
            rf"\b(?:crypto\s*\.\s*)?randomBytes\s*\([^;\n]*\)"
            rf"\s*\.\s*copy\s*\(\s*{name}\b",
            "secure_random",
        ),
        (
            rf"\bRandomNumberGenerator\s*\.\s*Fill\s*\(\s*{name}\b",
            "secure_random",
        ),
        (
            rf"\b(?:new\s+(?:System\.)?Random\s*\(.*?\)|"
            rf"[A-Za-z_]\w*)\s*\.\s*NextBytes\s*\(\s*{name}\b",
            "non_csprng",
        ),
    )
    for pattern, origin in patterns:
        matches = tuple(
            match
            for match in re.finditer(pattern, prefix, flags=re.DOTALL)
            if analyzer.position_is_in_same_or_enclosing_function(
                match.start(),
                start,
            )
        )
        if matches:
            candidates.append((matches[-1].start(), origin))

    for match in re.finditer(
        rf"\b(?P<receiver>[A-Za-z_]\w*)\s*\.\s*(?:GetBytes|Fill)\s*\(\s*{name}\b",
        prefix,
    ):
        if not analyzer.position_is_in_same_or_enclosing_function(
            match.start(),
            start,
        ):
            continue
        receiver = match.group("receiver")
        if re.search(
            rf"\b(?:RandomNumberGenerator|RNGCryptoServiceProvider)\s+"
            rf"{re.escape(receiver)}\b",
            prefix[: match.start()],
        ):
            candidates.append((match.start(), "secure_random"))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _detect_java_f9_m05(
    analyzer: JavaAnalyzer,
    task_id: str,
) -> DetectionResult:
    """Resolve the task-designated Java random value instead of PRNG objects."""

    source = analyzer.source
    flow = analyzer.flow
    findings: list[DetectorFinding] = []
    sink_specs: tuple[tuple[str, int, str], ...]
    if task_id == "F9_T1":
        sink_specs = (
            ("GCMParameterSpec", 1, "iv"),
            ("IvParameterSpec", 0, "iv"),
            ("ChaCha20ParameterSpec", 0, "iv"),
        )
    elif task_id == "F9_T2":
        sink_specs = (
            ("PBEKeySpec", 1, "salt"),
            ("PBEParameterSpec", 0, "salt"),
        )
    else:
        sink_specs = ()

    if task_id == "F9_T3":
        generator_findings = _java_f9_called_generator_findings(analyzer)
        if generator_findings:
            return combine_detector_findings(
                "M05",
                "java",
                _dedupe_findings(generator_findings),
                no_sink_reason=(
                    "no generated randomness reaches the task-designated output"
                ),
            )
        output_findings = _java_f9_output_findings(analyzer)
        if output_findings:
            return combine_detector_findings(
                "M05",
                "java",
                _dedupe_findings(output_findings),
                no_sink_reason=(
                    "no generated randomness reaches the task-designated output"
                ),
            )

    for class_name, argument_index, role in sink_specs:
        for call in constructor_calls(source, class_name):
            if len(call.arguments) <= argument_index:
                continue
            if task_id == "F9_T1" and _java_f9_parameter_spec_is_decryption_reuse(
                source,
                call.start,
                call.end,
            ):
                continue
            argument = call.arguments[argument_index]
            classification = _classify_java_f9_origin(
                flow,
                argument.text,
                argument.start,
                "random_value",
            )
            findings.append(
                _java_m05_finding(
                    source,
                    sink_name=f"new {class_name}",
                    expression=argument.text,
                    start=argument.start,
                    end=argument.end,
                    label=classification.label,
                    reason=classification.reason,
                )
            )

    if task_id == "F9_T3":
        byte_helper_findings: list[DetectorFinding] = []
        draw_helpers = tuple(
            method for method in flow.methods if method.name == "drawRandomBytes"
        )
        for call in flow._method_calls("drawRandomBytes"):
            if any(
                method.start <= call.start <= method.body_start
                for method in draw_helpers
            ):
                continue
            if len(call.arguments) < 2:
                continue
            expression = source[call.start : call.end]
            classification = _classify_java_f9_origin(
                flow,
                expression,
                call.start,
                "random_value",
            )
            if classification.label == "unknown":
                continue
            byte_helper_findings.append(
                _java_m05_finding(
                    source,
                    sink_name="drawRandomBytes",
                    expression=expression,
                    start=call.start,
                    end=call.end,
                    label=classification.label,
                    reason=classification.reason,
                )
            )
        if byte_helper_findings:
            return combine_detector_findings(
                "M05",
                "java",
                _dedupe_findings(byte_helper_findings),
                no_sink_reason="no generated byte challenge reaches the task-designated output",
            )
        designated_call_findings: list[DetectorFinding] = []
        for method in flow.methods:
            if re.search(
                r"(?i)^(?:run|execute|process|use|handle|with).*"
                r"(?:challenge|random|operation|security|token)",
                method.name,
            ) is None:
                continue
            parameters = [parameter.name for parameter in method.params]
            role_indexes = [
                parameter_index
                for parameter_index, parameter in enumerate(parameters)
                if _F9_RANDOM_OUTPUT_ROLE.fullmatch(parameter) is not None
            ]
            for call in flow._method_calls(method.name):
                if not flow._call_may_target_declared_method(call, method):
                    continue
                if method.start <= call.start <= method.body_end:
                    continue
                for parameter_index in role_indexes:
                    if parameter_index >= len(call.arguments):
                        continue
                    argument = call.arguments[parameter_index]
                    classification = _classify_java_f9_origin(
                        flow,
                        argument.text,
                        argument.start,
                        "random_value",
                    )
                    if classification.label == "unknown":
                        continue
                    designated_call_findings.append(
                        _java_m05_finding(
                            source,
                            sink_name=method.name,
                            expression=argument.text,
                            start=argument.start,
                            end=argument.end,
                            label=classification.label,
                            reason=classification.reason,
                        )
                    )
        if designated_call_findings:
            findings.extend(designated_call_findings)
        else:
            assignment_findings: list[tuple[DetectorFinding, bool]] = []
            for assignment in flow.assignments:
                byte_role = _java_f9_assignment_is_byte_role(
                    source,
                    assignment.name,
                    assignment.start,
                )
                if not (
                    _java_f9_assignment_is_integer_role(
                        source,
                        assignment.name,
                        assignment.start,
                    )
                    or byte_role
                ):
                    continue
                direct_byte_allocation = byte_role and re.fullmatch(
                    r"new\s+byte\s*\[[^\]]+\]",
                    assignment.expression.strip(),
                ) is not None
                containing_method = next(
                    (
                        method
                        for method in flow.methods
                        if method.body_start < assignment.start < method.body_end
                    ),
                    None,
                )
                classification = _classify_java_f9_origin(
                    flow,
                    (
                        assignment.name
                        if direct_byte_allocation
                        else assignment.expression
                    ),
                    (
                        containing_method.body_end
                        if direct_byte_allocation and containing_method is not None
                        else assignment.start
                    ),
                    "random_value",
                )
                if classification.label == "unknown":
                    continue
                assignment_findings.append(
                    (
                        _java_m05_finding(
                            source,
                            sink_name=assignment.name,
                            expression=assignment.expression,
                            start=assignment.start,
                            end=assignment.end,
                            label=classification.label,
                            reason=classification.reason,
                        ),
                        _java_f9_has_explicit_origin_evidence(
                            flow,
                            assignment.expression,
                            assignment.start,
                        ),
                    )
                )

            # SecurityValueFlow intentionally focuses on declarations and can
            # omit a later assignment such as ``value = random.nextInt()``.
            # Recover those numeric assignments, which are common in bounded
            # rejection-sampling blocks, without treating downstream challenge
            # objects or formatted tokens as fresh random decisions.
            for match in re.finditer(
                r"(?m)(?<![\w.])(?P<name>[A-Za-z_]\w*)\s*=\s*"
                r"(?P<expression>[^;\n]+);",
                source,
            ):
                name = match.group("name")
                expression = match.group("expression").strip()
                if not (
                    _F9_RANDOM_OUTPUT_ROLE.fullmatch(name) is not None
                    or name.lower() == "value"
                ):
                    continue
                if re.search(
                    rf"\b(?:byte|short|int|long|Integer|Long)\s+"
                    rf"{re.escape(name)}\b",
                    source[: match.start("name")],
                ) is None:
                    continue
                if not _java_f9_has_explicit_origin_evidence(
                    flow,
                    expression,
                    match.start("expression"),
                ):
                    continue
                classification = _classify_java_f9_origin(
                    flow,
                    expression,
                    match.start("expression"),
                    "random_value",
                )
                if classification.label == "unknown":
                    continue
                assignment_findings.append(
                    (
                        _java_m05_finding(
                            source,
                            sink_name=name,
                            expression=expression,
                            start=match.start("expression"),
                            end=match.end("expression"),
                            label=classification.label,
                            reason=classification.reason,
                        ),
                        True,
                    )
                )

            if assignment_findings:
                direct = [
                    finding
                    for finding, has_evidence in assignment_findings
                    if has_evidence
                ]
                findings.extend(
                    direct or [finding for finding, _ in assignment_findings]
                )
        if not findings:
            for method in flow.methods:
                if re.search(
                    r"(?i)^(?:run|execute|process|use|handle|with).*"
                    r"(?:challenge|random|operation|security|token)",
                    method.name,
                ) is None:
                    continue
                parameters = [parameter.name for parameter in method.params]
                role_indexes = [
                    parameter_index
                    for parameter_index, parameter in enumerate(parameters)
                    if _F9_RANDOM_OUTPUT_ROLE.fullmatch(parameter) is not None
                ]
                for call in flow._method_calls(method.name):
                    if not flow._call_may_target_declared_method(call, method):
                        continue
                    if method.start <= call.start <= method.body_end:
                        continue
                    for parameter_index in role_indexes:
                        if parameter_index >= len(call.arguments):
                            continue
                        argument = call.arguments[parameter_index]
                        classification = _classify_java_f9_origin(
                            flow,
                            argument.text,
                            argument.start,
                            "random_value",
                        )
                        if classification.label == "unknown":
                            continue
                        findings.append(
                            _java_m05_finding(
                                source,
                                sink_name=method.name,
                                expression=argument.text,
                                start=argument.start,
                                end=argument.end,
                                label=classification.label,
                                reason=classification.reason,
                            )
                        )

    if not findings:
        return _detect_java_m05_standard(analyzer)
    findings = [
        finding
        for finding in findings
        if finding.span.start < 0
        or analyzer.position_is_reachable(finding.span.start)
    ]
    return combine_detector_findings(
        "M05",
        "java",
        _dedupe_findings(findings),
        no_sink_reason="no generated randomness reaches the task-designated security sink",
    )


def _java_f9_called_generator_findings(
    analyzer: JavaAnalyzer,
) -> list[DetectorFinding]:
    """Return evidence owned by an invoked challenge-generator helper."""

    source = analyzer.source
    flow = analyzer.flow
    findings: list[DetectorFinding] = []
    for method in flow.methods:
        header = source[method.start : method.body_start]
        integer_return = bool(
            re.search(
                rf"\b(?:byte|short|int|long)\s+"
                rf"{re.escape(method.name)}\s*\(",
                header,
            )
        )
        if not _is_f9_integer_generator_name(method.name) or not integer_return:
            continue
        calls = tuple(
            call
            for call in flow._method_calls(method.name)
            if flow._call_may_target_declared_method(call, method)
            and not method.start <= call.start <= method.body_end
        )
        if not calls:
            continue
        body = source[method.body_start + 1 : method.body_end]
        if re.search(
            r"(?:\bSecureRandom\b|\b(?:java\.util\.)?Random\b|"
            r"\bMath\s*\.\s*random\s*\(|"
            r"\bSystem\s*\.\s*(?:currentTimeMillis|nanoTime)\s*\()",
            body,
        ) is None:
            continue
        for returned in re.finditer(r"\breturn\s+(?P<value>[^;]+);", body):
            start = method.body_start + 1 + returned.start("value")
            expression = returned.group("value").strip()
            classification = _classify_java_f9_origin(
                flow,
                expression,
                start,
                "random_value",
            )
            if classification.label == "unknown":
                continue
            findings.append(
                _java_m05_finding(
                    source,
                    sink_name=method.name,
                    expression=expression,
                    start=start,
                    end=start + len(expression),
                    label=classification.label,
                    reason=classification.reason,
                )
            )
    return findings


def _java_f9_output_findings(
    analyzer: JavaAnalyzer,
) -> list[DetectorFinding]:
    """Classify the random value that a Java F9_T3 program outputs.

    A program may construct several generators while only one controls its
    output.  Following ``print`` and ``println`` arguments avoids treating an
    unused decoy generator as the task result and naturally preserves mixed
    branches when different generators can reach the output.
    """

    findings: list[DetectorFinding] = []
    for method_name in ("print", "println"):
        for call in method_calls(analyzer.source, method_name):
            if not analyzer.position_is_reachable(call.start):
                continue
            for argument in call.arguments:
                if re.fullmatch(
                    r"[A-Za-z_]\w*\s*\.\s*[A-Za-z_]\w*",
                    argument.text.strip(),
                ) and analyzer.flow.structured_member_sources(
                    argument.text,
                    argument.start,
                ):
                    # Prefer the callsite that supplied a challenge to a
                    # record over its later presentation from that record.
                    # This keeps the security decision attached to the
                    # generation operation rather than a transport field.
                    continue
                evidence = _java_f9_origin_evidence(
                    analyzer.flow,
                    argument.text,
                    argument.start,
                )
                if re.search(
                    r"\b(?:generateSeed|nextBytes|nextInt|nextLong)\s*\(|"
                    r"\bMath\s*\.\s*random\s*\(",
                    evidence,
                ) is None:
                    continue
                classification = _classify_java_f9_origin(
                    analyzer.flow,
                    argument.text,
                    argument.start,
                    "random_value",
                )
                if classification.label == "unknown":
                    continue
                findings.append(
                    _java_m05_finding(
                        analyzer.source,
                        sink_name=f"System.out.{method_name}",
                        expression=argument.text,
                        start=argument.start,
                        end=argument.end,
                        label=classification.label,
                        reason=classification.reason,
                    )
                )
    return findings


def _java_f9_assignment_is_integer_role(
    source: str,
    name: str,
    start: int,
) -> bool:
    line_start = source.rfind("\n", 0, start) + 1
    line_end = source.find("\n", start)
    if line_end == -1:
        line_end = len(source)
    declaration = source[line_start:line_end]
    role_name = bool(
        _F9_RANDOM_OUTPUT_ROLE.fullmatch(name)
        or _F9_STRONG_INTEGER_OUTPUT_ROLE.fullmatch(name)
    )
    direct_value_source = bool(
        name.lower() == "value"
        and re.search(
            r"(?:\bSecureRandom\b|\b(?:java\.util\.)?Random\b|"
            r"\bMath\s*\.\s*random\s*\(|"
            r"\bSystem\s*\.\s*(?:currentTimeMillis|nanoTime)\s*\()",
            declaration,
        )
    )
    if not (role_name or direct_value_source):
        return False
    numeric_declaration = re.search(
        rf"\b(?:byte|short|int|long|Integer|Long)\s+"
        rf"{re.escape(name)}\b",
        declaration,
    )
    if numeric_declaration is not None:
        return True
    return bool(
        role_name
        and re.search(
            rf"\b(?:byte|short|int|long|Integer|Long)\s+"
            rf"{re.escape(name)}\b",
            source[:start],
        )
    )


def _java_f9_assignment_is_byte_role(
    source: str,
    name: str,
    start: int,
) -> bool:
    if _F9_RANDOM_OUTPUT_ROLE.fullmatch(name) is None:
        return False
    line_start = source.rfind("\n", 0, start) + 1
    line_end = source.find("\n", start)
    if line_end == -1:
        line_end = len(source)
    declaration = source[line_start:line_end]
    return re.search(
        rf"\bbyte\s*\[\s*\]\s+{re.escape(name)}\b",
        declaration,
    ) is not None


def _classify_java_f9_origin(
    flow: SecurityValueFlow,
    expression: str,
    index: int,
    role: str,
) -> SourceClassification:
    classification = flow.classify(expression, index, role)
    if classification.label == "mixed":
        return classification
    seeded_receiver = re.match(
        r"\s*(?P<name>[A-Za-z_]\w*)\s*\.\s*"
        r"(?:generateSeed|nextBytes|nextInt|nextLong)\s*\(",
        expression,
    )
    if seeded_receiver is not None:
        seed_quality = _java_prng_receiver_seed_quality(
            flow,
            seeded_receiver.group("name"),
            index,
        )
        if seed_quality is not None:
            return seed_quality

    # Arithmetic used to bound a generated integer is deterministic but does
    # not make the underlying PRNG predictable. Classify all random origins in
    # the controlled expression before falling back to generic constant flow.
    evidence = _java_f9_origin_evidence(flow, expression, index)
    controlled_evidence = f"{expression}\n{evidence}"
    has_secure_origin = re.search(
        r"\b(?:new\s+(?:java\.security\.)?SecureRandom\s*\(|"
        r"(?:java\.security\.)?SecureRandom\s*\.\s*getInstance)",
        controlled_evidence,
    ) is not None and re.search(
        r"\b(?:generateSeed|nextBytes|nextInt|nextLong)\s*\(",
        controlled_evidence,
    ) is not None
    has_weak_origin = re.search(
        r"\b(?:new\s+(?:java\.util\.)?Random\s*\(|Math\s*\.\s*random\s*\()",
        controlled_evidence,
    ) is not None
    if has_secure_origin and has_weak_origin:
        return SourceClassification(
            "mixed",
            f"{role} can be generated by both SecureRandom and a non-CSPRNG",
        )
    if has_secure_origin:
        return SourceClassification(
            "secure",
            f"{role} is generated by SecureRandom",
        )
    if has_weak_origin:
        return SourceClassification(
            "misuse",
            f"{role} is generated by a non-CSPRNG or clock value",
        )
    if classification.label != "unknown":
        return classification
    return classification


def _java_f9_origin_evidence(
    flow: SecurityValueFlow,
    expression: str,
    index: int,
    seen: frozenset[tuple[str, int]] = frozenset(),
) -> str:
    return random_origin_evidence(flow, expression, index, seen)


def _java_f9_has_explicit_origin_evidence(
    flow: SecurityValueFlow,
    expression: str,
    index: int,
) -> bool:
    return has_explicit_random_origin(flow, expression, index)


def _java_prng_receiver_seed_quality(
    flow: SecurityValueFlow,
    receiver: str,
    index: int,
) -> SourceClassification | None:
    """Classify an explicit seed that controls a Java PRNG receiver."""

    prefix = flow.source[:index]
    seed_expressions: list[tuple[int, str]] = []
    for match in re.finditer(
        rf"\b{re.escape(receiver)}\s*\.\s*setSeed\s*\(\s*(?P<seed>[^;)]+)\s*\)",
        prefix,
    ):
        seed_expressions.append((match.start(), match.group("seed")))
    for match in re.finditer(
        rf"\b(?:java\.security\.)?SecureRandom\s+{re.escape(receiver)}\s*=\s*"
        rf"new\s+(?:java\.security\.)?SecureRandom\s*\(\s*(?P<seed>[^)]+)\)",
        prefix,
    ):
        seed_expressions.append((match.start(), match.group("seed")))
    if not seed_expressions:
        return None

    _, seed = max(seed_expressions, key=lambda item: item[0])
    classified = flow.classify(seed, index, "prng_seed")
    if classified.label == "secure":
        return SourceClassification(
            "secure",
            "random_value is controlled by a fresh CSPRNG seed",
        )
    if classified.label in {"misuse", "mixed"}:
        return SourceClassification(
            classified.label,
            "random_value is controlled by a constant or predictable seed",
        )
    return SourceClassification(
        "unknown",
        "random_value seed origin could not be resolved",
    )


def _java_m05_finding(
    source: str,
    *,
    sink_name: str,
    expression: str,
    start: int,
    end: int,
    label: ProgramLabel,
    reason: str,
) -> DetectorFinding:
    line, column = _line_column(source, start)
    reason_code = {
        "secure": "approved_csprng",
        "misuse": "non_csprng_or_predictable_seed",
        "mixed": "mixed_randomness_origin",
        "unknown": "unresolved_randomness_origin",
    }[label]
    return DetectorFinding(
        misuse_id="M05",
        language="java",
        sink_id="security_random_source",
        sink_name=sink_name,
        expression=expression,
        label=label,
        reason_code=reason_code,
        reason=reason,
        policy_id="M05.policy.v1",
        span=SourceSpan(start=start, end=end, line=line, column=column),
    )



def _detect_java_m05(
    analyzer: JavaAnalyzer,
    task_id: str | None,
) -> DetectionResult:
    if task_id in {"F9_T1", "F9_T2", "F9_T3"}:
        return _detect_java_f9_m05(analyzer, task_id)
    return _detect_java_m05_standard(analyzer)


def _detect_java_m05_standard(analyzer: JavaAnalyzer) -> DetectionResult:
    source = analyzer.source
    no_sink_reason = "no registered security-sensitive randomness use found"
    if not source_contains_any_token(
        source,
        {
            "SecureRandom",
            "Random",
            "setSeed",
            "getInstanceStrong",
            "getInstance",
        },
    ):
        return evidence_result(
            analyzer,
            "M05",
            (),
            no_sink_reason=no_sink_reason,
        )
    if not analyzer.flow.ast_context.parse_success:
        return parse_failure_result(analyzer, "M05")
    if not _java_prng_security_context(source):
        return evidence_result(
            analyzer,
            "M05",
            (),
            no_sink_reason=no_sink_reason,
        )

    evidence: list[JavaEvidence] = []
    for class_name, sink_name in (
        ("SecureRandom", "new SecureRandom seed"),
    ):
        for call in constructor_calls(source, class_name):
            if not call.arguments:
                continue
            argument = call.arguments[0]
            evidence.append(
                _java_prng_seed_evidence(
                    analyzer,
                    sink_name,
                    argument.text,
                    argument.start,
                    argument.end,
                )
            )
    for match in re.finditer(
        r"\bnew\s+(?:java\s*\.\s*util\s*\.\s*)?Random\s*\(",
        source,
    ):
        evidence.append(
            JavaEvidence(
                sink_name="java.util.Random",
                expression=source[match.start() : match.end()],
                label="misuse",
                reason="java.util.Random is not a cryptographic random generator",
                start=match.start(),
                end=match.end(),
            )
        )
    for match in re.finditer(r"\bMath\s*\.\s*random\s*\(", source):
        evidence.append(
            JavaEvidence(
                sink_name="Math.random",
                expression=source[match.start() : match.end()],
                label="misuse",
                reason="Math.random is not a cryptographic random generator",
                start=match.start(),
                end=match.end(),
            )
        )
    for call in method_calls(source, "setSeed"):
        if not call.arguments:
            continue
        argument = call.arguments[0]
        evidence.append(
            _java_prng_seed_evidence(
                analyzer,
                "setSeed",
                argument.text,
                argument.start,
                argument.end,
            )
        )
    if not evidence:
        evidence.extend(_java_unseeded_secure_random_evidence(source))
    return evidence_result(
        analyzer,
        "M05",
        evidence,
        no_sink_reason=no_sink_reason,
    )


def _java_prng_seed_evidence(
    analyzer: JavaAnalyzer,
    sink_name: str,
    expression: str,
    start: int,
    end: int,
) -> JavaEvidence:
    classification = analyzer.flow.classify(
        expression,
        start,
        "prng_seed",
    )
    return JavaEvidence(
        sink_name=sink_name,
        expression=expression,
        label=classification.label,
        reason=classification.reason,
        start=start,
        end=end,
    )


def _java_prng_security_context(source: str) -> bool:
    if not re.search(
        r"\b(?:SecureRandom|Random)\b|\bMath\s*\.\s*random\b",
        source,
    ):
        return False
    if re.search(
        r"\b(?:nextBytes|nextInt|nextLong|generateSeed|KeyGenerator|SecretKeySpec|"
        r"IvParameterSpec|PBEKeySpec|PBEParameterSpec|nonce|iv|salt|key|"
        r"token|password)\b",
        source,
        flags=re.IGNORECASE,
    ):
        return True
    return security_context_present(source)


def _java_unseeded_secure_random_evidence(
    source: str,
) -> list[JavaEvidence]:
    evidence: list[JavaEvidence] = []
    for call in constructor_calls(source, "SecureRandom"):
        if call.arguments:
            continue
        evidence.append(
            JavaEvidence(
                sink_name="new SecureRandom without explicit seed",
                expression="new SecureRandom()",
                label="secure",
                reason=(
                    "SecureRandom is used without an explicit predictable seed"
                ),
                start=call.start,
                end=call.end,
            )
        )
    for method_name, sink_name, reason in (
        (
            "getInstanceStrong",
            "SecureRandom.getInstanceStrong",
            "SecureRandom.getInstanceStrong is used without an explicit predictable seed",
        ),
        (
            "getInstance",
            "SecureRandom.getInstance",
            "SecureRandom.getInstance is used without an explicit predictable seed",
        ),
    ):
        for call in static_method_calls(source, "SecureRandom", method_name):
            evidence.append(
                JavaEvidence(
                    sink_name=sink_name,
                    expression=source[call.start : call.end],
                    label="secure",
                    reason=reason,
                    start=call.start,
                    end=call.end,
                )
            )
    return evidence


MISUSE_ID = "M05"


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
        implementation=_detect_m05,
    )


__all__ = ("MISUSE_ID", "detect")


def _prioritize_f9_t3_output_relevance(
    analyzer: SourceAnalyzer,
    findings: list[DetectorFinding],
) -> list[DetectorFinding]:
    """Drop random calls that do not determine the requested integer output."""

    if len(findings) < 2:
        return findings
    scores = {
        id(finding): _f9_t3_finding_output_score(analyzer, finding)
        for finding in findings
    }
    maximum = max(scores.values(), default=0)
    if maximum <= 0:
        return findings
    return [finding for finding in findings if scores[id(finding)] == maximum]


def _f9_t3_finding_output_score(
    analyzer: SourceAnalyzer,
    finding: DetectorFinding,
) -> int:
    start = finding.span.start
    statement_start = max(
        analyzer.source.rfind(";", 0, start),
        analyzer.source.rfind("\n", 0, start),
        analyzer.source.rfind("{", 0, start),
    ) + 1
    statement_end = analyzer.source.find(";", finding.span.end)
    if statement_end == -1:
        statement_end = min(len(analyzer.source), finding.span.end + 240)
    statement = analyzer.source[statement_start : statement_end + 1]
    score = 0

    assigned = re.search(
        r"(?:\b(?:const|let|var)\s+|\bself\.|\bthis\.)?"
        r"(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)"
        r"(?:\s*:[^=;]+)?\s*=",
        analyzer.source[statement_start:start],
    )
    if assigned is not None:
        name = assigned.group("name")
        score = max(score, _f9_output_role_priority(name))
        tail = analyzer.source[statement_end + 1 :]
        containing_name = analyzer.containing_function_name(start)
        if containing_name:
            definitions = tuple(
                definition
                for definition in getattr(analyzer, "functions", ())
                if definition.name == containing_name
                and definition.start <= start <= definition.end
            )
            if definitions:
                tail = analyzer.source[statement_end + 1 : definitions[0].end]
        if re.search(
            rf"\b(?:return\s+|(?:console\s*\.\s*log|print)\s*\([^)]*)"
            rf"{re.escape(name)}\b",
            tail,
        ):
            score += 10

    prefix = analyzer.source[statement_start:start]
    if re.search(r"\breturn\b|\b(?:console\s*\.\s*log|print)\s*\(", prefix):
        score += 10
    function_name = analyzer.containing_function_name(start) or ""
    if function_name.lower() not in {"constructor", "__init__"} and re.search(
        r"(?i)(?:generate|create|draw|next|issue|make).*(?:number|integer|code|challenge)"
        r"|(?:number|integer|code|challenge).*(?:generate|create|draw|next|issue|make)",
        function_name,
    ):
        score += 8
    return score


def _legacy_f9_random_source_findings(
    analyzer: SourceAnalyzer,
) -> list[DetectorFinding]:
    """Retain the seed-oriented F9 contract used by the earlier dataset."""

    candidates = {
        "typescript": (
            "randomBytes", "randomFill", "randomFillSync", "randomInt",
            "getRandomValues", "random", "imul", "m05CreateBytes",
            "m05RandomSource",
        ),
        "csharp": (
            "GetBytes", "Fill", "GetInt32", "NextBytes", "Next",
            "NextInt64", "m05CreateBytes", "m05RandomSource",
        ),
        "python": (
            "urandom", "token_bytes", "randbits", "randbelow", "randbytes",
            "randint", "randrange", "_m05_random_source",
            "_m05_create_bytes",
        ),
    }[analyzer.language]
    findings: list[DetectorFinding] = []
    for call in analyzer.calls(*candidates):
        if not _is_random_source_call(analyzer, call):
            continue
        expression = analyzer.source[call.start : call.end]
        api_origin = _random_source_api_origin(analyzer, call)
        resolved = (
            ResolvedValue(expression, api_origin)
            if api_origin is not None
            else _resolve_f9_random_origin(analyzer, expression, call.start)
        )
        findings.append(
            _finding_from_resolved(
                analyzer,
                "M05",
                "security_random_source",
                call.callee,
                expression,
                call.start,
                call.end,
                resolved,
                _random_label,
            )
        )
    if analyzer.language == "typescript":
        findings = _suppress_custom_prng_seed_findings(analyzer, findings)
    return _dedupe_findings(findings)


def _java_f9_parameter_spec_is_decryption_reuse(
    source: str,
    start: int,
    end: int,
) -> bool:
    """Return whether a Java IV/nonce spec first reaches a decrypt init."""

    statement_start = max(source.rfind(";", 0, start), source.rfind("{", 0, start)) + 1
    statement_end = source.find(";", end)
    if statement_end == -1:
        statement_end = min(len(source), end + 400)
    statement = source[statement_start : statement_end + 1]
    if "DECRYPT_MODE" in statement:
        return True

    assigned = re.search(
        r"\b(?P<name>[A-Za-z_]\w*)\s*=\s*new\s+"
        r"(?:GCMParameterSpec|IvParameterSpec|ChaCha20ParameterSpec)\s*\(",
        statement,
    )
    if assigned is None:
        return False
    name = assigned.group("name")
    use = re.search(
        rf"\binit\s*\(\s*Cipher\.(?P<mode>ENCRYPT_MODE|DECRYPT_MODE)"
        rf"[^;]*\b{re.escape(name)}\b[^;]*\);",
        source[statement_end + 1 :],
        flags=re.DOTALL,
    )
    return use is not None and use.group("mode") == "DECRYPT_MODE"


__all__ = ("MISUSE_ID", "detect")
