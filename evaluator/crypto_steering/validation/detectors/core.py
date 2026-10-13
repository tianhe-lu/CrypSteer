"""Finding construction, classification, filtering, and detector utilities."""

from __future__ import annotations

import ast
from dataclasses import replace
import re
from typing import Callable, Iterable

from crypto_steering.validation.models import (
    DetectionResult,
    DetectorFinding,
    ProgramLabel,
    ResolvedValue,
    SourceSpan,
    ValueOrigin,
    combine_detector_findings,
)
from crypto_steering.validation.source.index import (
    CStyleAnalyzer,
    CallExpression,
    FunctionDefinition,
    PythonAnalyzer,
    SourceAnalyzer,
    csharp_activator_constructor_arguments,
    mask_c_style_comments_and_literals,
    split_arguments,
)

_SECURE_DIGESTS = {
    "SHA224", "SHA256", "SHA384", "SHA512",
    "SHA3224", "SHA3256", "SHA3384", "SHA3512",
    "BLAKE2B256", "BLAKE2B512", "BLAKE2S256",
}
_BROKEN_DIGESTS = {"MD2", "MD4", "MD5", "SHA", "SHA1"}
_SECURE_MACS = {
    "HMACSHA224", "HMACSHA256", "HMACSHA384", "HMACSHA512",
    "HMACSHA3224", "HMACSHA3256", "HMACSHA3384", "HMACSHA3512",
    "HMACBLAKE2B256", "HMACBLAKE2B512", "HMACBLAKE2S256",
    "AESCMAC", "KMAC128", "KMAC256",
}
_BROKEN_MACS = {"HMACMD5", "HMACSHA", "HMACSHA1", "PBEWITHHMACSHA1"}


def _origin_call_findings(
    analyzer: SourceAnalyzer,
    misuse_id: str,
    specs: Iterable[tuple[str, int]],
    sink_id: str,
    classifier: Callable[[ResolvedValue], tuple[ProgramLabel, str, str]],
) -> list[DetectorFinding]:
    findings: list[DetectorFinding] = []
    for name, argument_index in specs:
        for call in analyzer.calls(name):
            if (
                analyzer.language == "csharp"
                and len(call.arguments) > argument_index
                and re.match(r"\s*out\b", call.arguments[argument_index])
            ):
                # An out argument receives a value from the helper; it is not
                # key material consumed by that helper call.
                continue
            expression = _call_argument(call, argument_index)
            if expression is None:
                continue
            findings.append(_origin_finding(analyzer, misuse_id, call, expression, sink_id, classifier))
    return _dedupe_findings(findings)


def _origin_finding(
    analyzer: SourceAnalyzer,
    misuse_id: str,
    call: CallExpression,
    expression: str,
    sink_id: str,
    classifier: Callable[[ResolvedValue], tuple[ProgramLabel, str, str]],
) -> DetectorFinding:
    expression = _argument_value(expression)
    resolved = analyzer.resolve_origin(expression, call.start)
    return _finding_from_resolved(
        analyzer, misuse_id, sink_id, call.callee, expression,
        call.start, call.end, resolved, classifier,
    )


def _finding_from_resolved(
    analyzer: SourceAnalyzer,
    misuse_id: str,
    sink_id: str,
    sink_name: str,
    expression: str,
    start: int,
    end: int,
    resolved: ResolvedValue,
    classifier: Callable[[ResolvedValue], tuple[ProgramLabel, str, str]],
) -> DetectorFinding:
    label, code, reason = classifier(resolved)
    return DetectorFinding(
        misuse_id=misuse_id,
        language=analyzer.language,
        sink_id=sink_id,
        sink_name=sink_name,
        expression=expression,
        label=label,
        reason_code=code,
        reason=reason,
        span=analyzer.source_span(start, end),
        resolved_values=resolved.values,
        value_origin=resolved.origin,
        resolution_steps=resolved.steps,
    )


def _algorithm_finding(
    analyzer: SourceAnalyzer,
    misuse_id: str,
    call: CallExpression,
    expression: str,
    resolved: ResolvedValue,
    classifier: Callable[[str], tuple[ProgramLabel, str, str]],
) -> DetectorFinding:
    expression = _argument_value(expression)
    classifications = [classifier(value) for value in resolved.values]
    has_unrestricted_runtime_fallback = bool(
        any(
            step.action == "resolve logical fallback"
            for step in resolved.steps
        )
        and any(
            step.result in {"runtime_input", "unknown"}
            or re.search(
                    r"\b(?:process\s*\.\s*(?:argv|env)|Deno\s*\.\s*env|"
                    r"Environment\s*\.\s*GetEnvironmentVariable|"
                    r"System\s*\.\s*getenv|os\s*\.\s*(?:environ|getenv)|"
                    r"sys\s*\.\s*argv)\b",
                    step.expression,
                )
            for step in resolved.steps
        )
    )
    has_runtime_selected_policy = _algorithm_resolution_has_runtime_input(
        analyzer,
        expression,
        call.start,
        resolved,
    )
    if has_unrestricted_runtime_fallback:
        label = "unknown"
        code = "runtime_algorithm_unresolved"
        reason = "runtime-selected algorithm is not constrained to an approved set"
    elif not classifications:
        label: ProgramLabel = "unknown"
        code = "unresolved_algorithm"
        reason = resolved.unresolved_reason or "algorithm could not be resolved"
    else:
        labels = {item[0] for item in classifications}
        if (
            "secure" in labels
            and "misuse" in labels
            and has_runtime_selected_policy
        ):
            label = "mixed"
            code = "runtime_algorithm_selection"
            reason = (
                "one target operation is constrained to both secure and "
                "misuse algorithms at runtime"
            )
        elif "secure" in labels and "misuse" in labels:
            label = "mixed"
            code = "multiple_algorithm_policies"
        elif "misuse" in labels:
            label = "misuse"
            code = next(item[1] for item in classifications if item[0] == "misuse")
        elif labels == {"secure"}:
            label = "secure"
            code = classifications[0][1]
        else:
            label = "unknown"
            code = "unclassified_algorithm"
        if code != "runtime_algorithm_selection":
            reason = "; ".join(dict.fromkeys(item[2] for item in classifications))
    return DetectorFinding(
        misuse_id=misuse_id,
        language=analyzer.language,
        sink_id=f"{misuse_id.lower()}_algorithm",
        sink_name=call.callee,
        expression=expression,
        label=label,
        reason_code=code,
        reason=reason,
        span=analyzer.source_span(call.start, call.end),
        resolved_values=resolved.values,
        value_origin=resolved.origin,
        resolution_steps=resolved.steps,
    )


def _contains_classified_algorithm(
    resolved: ResolvedValue,
    classifier: Callable[[str], tuple[ProgramLabel, str, str]],
) -> bool:
    return bool(resolved.values) and all(
        classifier(value)[0] in {"secure", "misuse"}
        for value in resolved.values
    )


def _resolved_is_explicit_mac_algorithm(resolved: ResolvedValue) -> bool:
    return bool(resolved.values) and all(
        _normalize_algorithm(value).startswith(("HMAC", "PBEWITHHMAC", "KMAC"))
        or _normalize_algorithm(value) == "AESCMAC"
        for value in resolved.values
    )


def _csharp_selected_hash_dispatch_findings(
    analyzer: SourceAnalyzer,
) -> tuple[list[DetectorFinding], tuple[tuple[int, int], ...]]:
    """Resolve the selected path through local C# hash dispatch helpers.

    Generated programs commonly put all supported algorithms in one helper and
    select one with a caller-provided string or enum. Counting every dormant
    branch makes a secure SHA-256 call look mixed. We replace those branch-level
    findings only when the helper call's selector resolves at the callsite.
    """

    return _csharp_selected_algorithm_dispatch_findings(
        analyzer,
        misuse_id="M09",
        classifier=_classify_digest,
        factory_pattern=re.compile(
            r"\b(?:MD5|SHA1|SHA224|SHA256|SHA384|SHA512|"
            r"SHA3_256|SHA3_384|SHA3_512)\s*\.\s*Create\s*\("
            r"|\bHashAlgorithm\s*\.\s*Create\s*\("
            r"|\bCryptoConfig\s*\.\s*CreateFromName\s*\("
        ),
    )


def _csharp_selected_mac_dispatch_findings(
    analyzer: SourceAnalyzer,
) -> tuple[list[DetectorFinding], tuple[tuple[int, int], ...]]:
    return _csharp_selected_algorithm_dispatch_findings(
        analyzer,
        misuse_id="M10",
        classifier=_classify_mac,
        factory_pattern=re.compile(
            r"\bHMAC(?:MD5|SHA1|SHA224|SHA256|SHA384|SHA512|"
            r"SHA3_256|SHA3_384|SHA3_512)\s*\("
        ),
    )


def _csharp_selected_algorithm_dispatch_findings(
    analyzer: SourceAnalyzer,
    *,
    misuse_id: str,
    classifier: Callable[[str], tuple[ProgramLabel, str, str]],
    factory_pattern: re.Pattern[str],
) -> tuple[list[DetectorFinding], tuple[tuple[int, int], ...]]:
    if not isinstance(analyzer, CStyleAnalyzer):
        return [], ()

    findings: list[DetectorFinding] = []
    selected_spans: set[tuple[int, int]] = set()
    for definition in analyzer.functions:
        body = analyzer.source[definition.start : definition.end]
        if not factory_pattern.search(body):
            continue
        named_selector_indexes = [
            index
            for index, name in enumerate(definition.parameters)
            if _is_csharp_algorithm_selector_parameter(name)
        ]
        selector_indexes = list(named_selector_indexes)
        if not selector_indexes:
            continue

        for call in analyzer.calls(definition.name):
            if definition.start <= call.start < definition.end:
                continue
            for parameter_index in selector_indexes:
                parameter_name = definition.parameters[parameter_index]
                if not _csharp_dispatch_parameter_used(
                    analyzer,
                    definition,
                    parameter_name,
                ):
                    continue
                expression = _call_argument(
                    call,
                    parameter_index,
                    parameter_name,
                )
                if expression is None:
                    continue
                resolved = analyzer.resolve_strings(expression, call.start)
                if not _contains_classified_algorithm(resolved, classifier):
                    if expression.strip() in {"null", "default", "default!"}:
                        continue
                    dispatch_values = _csharp_dispatch_algorithm_values(
                        analyzer,
                        definition,
                        classifier,
                    )
                    if dispatch_values:
                        resolved = ResolvedValue(
                            expression,
                            "runtime_input",
                            dispatch_values,
                            unresolved_reason=(
                                "selector is runtime-selected within a bounded "
                                "local algorithm dispatch"
                            ),
                        )
                if not _contains_classified_algorithm(resolved, classifier):
                    continue
                findings.append(
                    _algorithm_finding(
                        analyzer,
                        misuse_id,
                        call,
                        expression,
                        resolved,
                        classifier,
                    )
                )
                selected_spans.add((definition.start, definition.end))
                break

    return _dedupe_findings(findings), tuple(sorted(selected_spans))


def _is_csharp_algorithm_selector_parameter(name: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
    if normalized in {"name", "type", "kind"}:
        return True
    return re.fullmatch(
        r"(?:selected|configured|requested|chosen)?"
        r"(?:hash|digest|algorithm|mac)"
        r"(?:algorithm|name|type|kind|identifier|id)?",
        normalized,
    ) is not None


def _csharp_local_helper_has_no_algorithm_selector(
    analyzer: SourceAnalyzer,
    function_name: str,
) -> bool:
    if not isinstance(analyzer, CStyleAnalyzer):
        return False
    definitions = tuple(
        definition
        for definition in analyzer.functions
        if definition.name == function_name
    )
    return bool(definitions) and all(
        not any(
            _is_csharp_algorithm_selector_parameter(parameter)
            for parameter in definition.parameters
        )
        for definition in definitions
    )


def _csharp_dispatch_parameter_used(
    analyzer: CStyleAnalyzer,
    definition: FunctionDefinition,
    parameter_name: str,
) -> bool:
    """Return whether a dispatch selector reaches the function body.

    A rewritten wrapper may retain its original selector parameter while its
    real factory sink is controlled by a zero-argument helper.  The parameter
    name in the declaration is not evidence that runtime input selects the
    digest.
    """

    definition_text = analyzer.masked[definition.start : definition.end]
    body_start = definition_text.find("{")
    if body_start < 0:
        body_start = definition_text.find("=>")
        if body_start < 0:
            return False
        body_start += 2
    else:
        body_start += 1
    return re.search(
        rf"\b{re.escape(parameter_name)}\b",
        definition_text[body_start:],
    ) is not None


def _csharp_dispatch_algorithm_values(
    analyzer: CStyleAnalyzer,
    definition: FunctionDefinition,
    classifier: Callable[[str], tuple[ProgramLabel, str, str]],
) -> tuple[str, ...]:
    """Collect policy algorithms from one source-defined dispatch body."""

    body = analyzer.masked[definition.start : definition.end]
    candidates = set(
        re.findall(
            r"\b(?:HMAC)?(?:MD5|SHA1|SHA224|SHA256|SHA384|SHA512|"
            r"SHA3_224|SHA3_256|SHA3_384|SHA3_512)\b",
            body,
            flags=re.IGNORECASE,
        )
    )
    return tuple(
        sorted(
            value
            for value in candidates
            if classifier(value)[0] in {"secure", "misuse"}
        )
    )


def _typescript_guarded_algorithm_values(
    analyzer: SourceAnalyzer,
    call: CallExpression,
    expression: str,
    classifier: Callable[[str], tuple[ProgramLabel, str, str]],
) -> tuple[str, ...]:
    """Resolve a runtime hash selector restricted to a finite TS union.

    A type alias alone is not a runtime security proof because TypeScript
    assertions can bypass it. The fallback therefore also requires a
    terminating guard whose checked value reaches the sink selector and which
    enumerates every member of the string union.
    """

    if not isinstance(analyzer, CStyleAnalyzer) or analyzer.language != "typescript":
        return ()
    target_name = _typescript_expression_name(expression)
    if target_name is None:
        return ()
    annotation = _typescript_expression_annotation(analyzer, expression, call.start)
    values = (
        _typescript_string_union_values(analyzer.source, annotation)
        if annotation is not None
        else ()
    )
    array_guard_values: tuple[str, ...] = ()
    if not values:
        array_guard_values = _typescript_terminating_array_guard_values(
            analyzer.source,
            target_name,
        )
        values = array_guard_values
    comparison_guard_values: tuple[str, ...] = ()
    if not values:
        comparison_guard_values = _typescript_terminating_comparison_guard_values(
            analyzer.source,
            target_name,
        )
        values = comparison_guard_values
    if not values or not all(
        classifier(value)[0] in {"secure", "misuse"}
        for value in values
    ):
        return ()
    if array_guard_values or comparison_guard_values:
        return values
    if not _typescript_has_terminating_union_guard(
        analyzer.source,
        values,
        target_name,
    ):
        return ()
    return values


def _typescript_expression_name(expression: str) -> str | None:
    compact = re.sub(r"\s+", "", _argument_value(expression))
    match = re.fullmatch(
        r"(?:(?:[A-Za-z_$][A-Za-z0-9_$]*)\.)*"
        r"(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)",
        compact,
    )
    return match.group("name") if match is not None else None


def _typescript_expression_annotation(
    analyzer: CStyleAnalyzer,
    expression: str,
    index: int,
) -> str | None:
    compact = re.sub(r"\s+", "", _argument_value(expression))
    match = re.fullmatch(
        r"(?:(?P<owner>[A-Za-z_$][A-Za-z0-9_$]*)\.)?"
        r"(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)",
        compact,
    )
    if match is None:
        return None
    name = match.group("name")
    annotation_pattern = re.compile(
        rf"\b{re.escape(name)}\s*\??\s*:\s*"
        r"(?P<type>"
        r"(?:[\"'][^\"']+[\"']\s*\|\s*)+[\"'][^\"']+[\"']"
        r"|[A-Za-z_$][A-Za-z0-9_$]*)"
    )
    containing = [
        definition
        for definition in analyzer.functions
        if definition.start <= index <= definition.end
    ]
    scopes = []
    if containing:
        definition = min(containing, key=lambda item: item.end - item.start)
        scopes.append(analyzer.source[definition.start:index])
    scopes.append(analyzer.source)
    for scope in scopes:
        matches = tuple(annotation_pattern.finditer(scope))
        if matches:
            return matches[-1].group("type").strip()
    return None


def _typescript_string_union_values(
    source: str,
    annotation: str,
) -> tuple[str, ...]:
    body = annotation.strip()
    if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", body):
        alias = re.search(
            rf"\btype\s+{re.escape(body)}\s*=\s*(?P<body>[^;]+);",
            source,
            flags=re.DOTALL,
        )
        if alias is None:
            return ()
        body = alias.group("body").strip()
    parts = tuple(item.strip() for item in body.split("|"))
    if len(parts) < 2:
        return ()
    values: list[str] = []
    for part in parts:
        literal = re.fullmatch(r"([\"'])(?P<value>[^\"']+)\1", part)
        if literal is None:
            return ()
        values.append(literal.group("value"))
    return tuple(dict.fromkeys(values))


def _typescript_has_terminating_union_guard(
    source: str,
    values: tuple[str, ...],
    target_name: str,
) -> bool:
    for match in re.finditer(r"\bif\s*\(", source):
        opening = source.find("(", match.start(), match.end())
        closing = _matching_c_style_parenthesis(source, opening)
        if closing == -1:
            continue
        condition = source[opening + 1 : closing]
        guarded_names = _typescript_guard_comparison_names(
            condition,
            values,
            negative=True,
        )
        following = source[closing + 1 : closing + 600]
        if guarded_names and _typescript_guard_branch_terminates(following):
            if any(
                _typescript_guard_reaches_target(source, name, target_name)
                for name in guarded_names
            ):
                return True
    return False


def _typescript_guard_comparison_names(
    condition: str,
    values: tuple[str, ...],
    *,
    negative: bool,
) -> frozenset[str]:
    operator_pattern = r"!==?" if negative else r"===?"
    required_joiner = "&&" if negative else "||"
    if len(values) > 1 and condition.count(required_joiner) < len(values) - 1:
        return frozenset()
    names_by_value: list[set[str]] = []
    for value in values:
        escaped = re.escape(value)
        names = {
            match.group("name")
            for match in re.finditer(
                rf"(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*"
                rf"{operator_pattern}\s*[\"']{escaped}[\"']",
                condition,
            )
        }
        names.update(
            match.group("name")
            for match in re.finditer(
                rf"[\"']{escaped}[\"']\s*{operator_pattern}\s*"
                rf"(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)",
                condition,
            )
        )
        if not names:
            return frozenset()
        names_by_value.append(names)
    common = set.intersection(*names_by_value)
    return frozenset(common)


def _typescript_guard_reaches_target(
    source: str,
    guarded_name: str,
    target_name: str,
) -> bool:
    if guarded_name == target_name:
        return True
    guarded = re.escape(guarded_name)
    target = re.escape(target_name)
    return bool(
        re.search(
            rf"(?<![\w$]){target}(?:\s*:\s*[^=;]+)?\s*=\s*"
            rf"{guarded}(?![\w$])",
            source,
        )
        or re.search(
            rf"(?<![\w$]){target}\s*:\s*{guarded}(?![\w$])",
            source,
        )
    )


def _typescript_guard_branch_terminates(source: str) -> bool:
    statement = source.lstrip()
    if statement.startswith("{"):
        closing = _matching_c_style_brace(statement, 0)
        if closing == -1:
            return False
        body = mask_c_style_comments_and_literals(statement[1:closing])
        depth = 0
        for token in re.finditer(
            r"[{}]|\b(?:throw|return)\b|"
            r"(?:fail|usage|usageAndExit|printUsageAndExit)\s*\(|"
            r"process\s*\.\s*exit\s*\(",
            body,
        ):
            if token.group(0) == "{":
                depth += 1
            elif token.group(0) == "}":
                depth = max(0, depth - 1)
            elif depth == 0:
                return True
        return False
    else:
        semicolon = statement.find(";")
        if semicolon != -1:
            statement = statement[: semicolon + 1]
    return bool(
        re.match(
            r"(?:throw\b|(?:fail|usage|usageAndExit|printUsageAndExit)\s*\("
            r"|process\s*\.\s*exit\s*\()",
            statement,
        )
    )


def _matching_c_style_brace(source: str, opening: int) -> int:
    if opening < 0 or opening >= len(source) or source[opening] != "{":
        return -1
    depth = 0
    quote = ""
    escaped = False
    for index in range(opening, len(source)):
        char = source[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char in {'"', "'", "`"}:
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return -1


def _matching_c_style_parenthesis(source: str, opening: int) -> int:
    if opening < 0 or opening >= len(source) or source[opening] != "(":
        return -1
    depth = 0
    quote = ""
    escaped = False
    index = opening
    while index < len(source):
        char = source[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            index += 1
            continue
        if char in {'"', "'", "`"}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def _integer_policy_finding(
    analyzer: SourceAnalyzer,
    misuse_id: str,
    call: CallExpression,
    expression: str,
    minimum: int,
    sink_id: str,
) -> DetectorFinding:
    expression = _argument_value(expression)
    resolved = analyzer.resolve_ints(expression, call.start)
    integers = _parse_int_values(resolved.values)
    guarded_values = (
        analyzer.integer_expression_guard_values(expression, call.start)
        if isinstance(analyzer, PythonAnalyzer)
        else ()
    )
    if guarded_values:
        guarded_integers = set(guarded_values)
        # The guard describes every value the helper permits, while a concrete
        # caller can select a narrower subset. Keep that call-site proof when
        # it exists; otherwise the complete allowlist remains the policy.
        caller_narrowed = any(
            step.action in {
                "resolve caller arguments",
                "resolve fixed caller before fail-closed whitelist",
            }
            for step in resolved.steps
        )
        integers = (
            integers & guarded_integers
            if caller_narrowed and integers
            else guarded_integers
        )
    has_unresolved_path = bool(
        resolved.origin == "unknown"
        or resolved.unresolved_reason
        or any(step.result == "unknown" for step in resolved.steps)
        or _numeric_policy_has_runtime_configuration(
            analyzer,
            expression,
            resolved,
        )
    )
    if guarded_values:
        has_unresolved_path = False
    guarded_minimum = bool(
        (
            isinstance(analyzer, PythonAnalyzer)
            and analyzer.integer_expression_has_minimum_guard(
                expression,
                call.start,
                minimum,
            )
        )
        or (
            isinstance(analyzer, CStyleAnalyzer)
            and _c_style_integer_expression_has_minimum_guard(
                analyzer,
                expression,
                call.start,
                minimum,
            )
        )
        or any(
            step.action == "apply numeric maximum lower bound"
            and re.fullmatch(r"[-+]?\d+(?:_\d+)*", step.result.strip())
            and int(step.result.replace("_", "")) >= minimum
            for step in resolved.steps
        )
    )
    if guarded_minimum:
        label = "secure"
        code = "guarded_minimum"
        reason = f"a terminating guard enforces minimum {minimum} before the sink"
    elif not integers or has_unresolved_path:
        label: ProgramLabel = "unknown"
        code = "unresolved_numeric_argument"
        reason = (
            "numeric policy argument has a reachable value path whose minimum "
            "could not be proven"
        )
    elif any(value < minimum for value in integers) and any(value >= minimum for value in integers):
        label = "mixed"
        code = "mixed_numeric_policy"
        reason = f"resolved values span both sides of minimum {minimum}: {sorted(integers)}"
    elif any(value < minimum for value in integers):
        label = "misuse"
        code = "below_minimum"
        reason = f"resolved value {min(integers)} is below minimum {minimum}"
    else:
        label = "secure"
        code = "meets_minimum"
        reason = f"all resolved values meet minimum {minimum}: {sorted(integers)}"
    return DetectorFinding(
        misuse_id=misuse_id,
        language=analyzer.language,
        sink_id=sink_id,
        sink_name=call.callee,
        expression=expression,
        label=label,
        reason_code=code,
        reason=reason,
        span=analyzer.source_span(call.start, call.end),
        resolved_values=tuple(str(item) for item in sorted(integers)),
        value_origin=resolved.origin,
        metadata={
            "minimum": minimum,
            "guarded_minimum": guarded_minimum,
            "has_unresolved_path": has_unresolved_path,
        },
        resolution_steps=resolved.steps,
    )


def _numeric_policy_has_runtime_configuration(
    analyzer: SourceAnalyzer,
    expression: str,
    resolved: ResolvedValue,
) -> bool:
    """Fail closed when a secure numeric default has an unbounded runtime path."""

    if resolved.origin == "runtime_input" or any(
        step.result == "runtime_input" for step in resolved.steps
    ):
        return True
    if (
        resolved.values
        and resolved.origin != "unknown"
        and not any(step.result == "unknown" for step in resolved.steps)
    ):
        # A property-shaped expression is not necessarily runtime-controlled.
        # Object literals and helper-returned records commonly carry a fully
        # resolved iteration constant even when unrelated JSON or stdin code
        # exists elsewhere in the program.
        return False
    if re.search(r"(?i)(?:iter(?:ation)?s?|rounds?|count)", expression) is None:
        return False
    if isinstance(analyzer, CStyleAnalyzer) and (
        re.search(
            r"(?i)\b[A-Za-z_$][A-Za-z0-9_$]*"
            r"(?:\s*\.\s*[A-Za-z_$][A-Za-z0-9_$]*)*\s*\.\s*"
            r"(?:iter(?:ation)?s?|rounds?|count)\b",
            expression,
        )
        or any(
            re.search(
                r"(?i)\b[A-Za-z_$][A-Za-z0-9_$]*"
                r"(?:\s*\.\s*[A-Za-z_$][A-Za-z0-9_$]*)*\s*\.\s*"
                r"(?:iter(?:ation)?s?|rounds?|count)\b",
                step.expression,
            )
            for step in resolved.steps
        )
    ):
        source = analyzer.source
        loads_record = bool(
            re.search(
                r"\b(?:JSON\s*\.\s*parse|JsonSerializer\s*\.\s*Deserialize)\s*\(",
                source,
            )
            and (
                re.search(
                    r"\b(?:readFileSync|readFile|ReadAllText|ReadAllBytes)\s*\(",
                    source,
                )
                or re.search(
                    r"\b(?:stdin|process\s*\.\s*argv|Console\s*\.\s*ReadLine)\b",
                    source,
                )
            )
        )
        if loads_record and not re.search(
            r"(?i)(?:iter(?:ation)?s?|rounds?|count)\s*<\s*"
            r"(?:1_?000|[2-9][0-9_]{3,})",
            source,
        ):
            return True
    if isinstance(analyzer, PythonAnalyzer) and re.search(
        r"(?i)\b(?:params?|config|record|payload)\s*\.\s*"
        r"(?:iter(?:ation)?s?|rounds?|count)\b",
        expression,
    ):
        source = analyzer.source
        loads_record = bool(
            re.search(r"\b(?:json\s*\.\s*loads|pickle\s*\.\s*loads?)\s*\(", source)
            and re.search(
                r"\b(?:open|read|read_bytes|read_text|read_binary)\s*\(",
                source,
            )
        )
        has_minimum_check = bool(
            re.search(
                r"(?i)(?:iter(?:ation)?s?|rounds?|count)\s*<\s*"
                r"(?:1_?000|[2-9][0-9_]{3,})",
                source,
            )
        )
        if loads_record and not has_minimum_check:
            return True
    runtime_markers = {
        "typescript": (
            r"\bprocess\s*\.\s*(?:argv|env)\b",
            r"\bDeno\s*\.\s*(?:args|env)\b",
        ),
        "csharp": (
            r"\bargs\s*\[",
            r"\bEnvironment\s*\.\s*GetEnvironmentVariable\b",
        ),
        "python": (
            r"\bsys\s*\.\s*argv\b",
            r"\bos\s*\.\s*(?:environ|getenv)\b",
            r"\badd_argument\s*\([^)]*--iterations\b",
        ),
    }.get(analyzer.language, ())
    resolved_path = "\n".join(
        (expression,) + tuple(step.expression for step in resolved.steps)
    )
    return any(
        re.search(pattern, resolved_path, re.DOTALL)
        for pattern in runtime_markers
    )


def _c_style_integer_expression_has_minimum_guard(
    analyzer: CStyleAnalyzer,
    expression: str,
    index: int,
    minimum: int,
) -> bool:
    """Recognize bounded fail-closed C#/TypeScript integer guards."""

    compact = re.sub(r"\s+", "", expression)
    if re.search(
        rf"\b(?:Math\.)?(?:max|Max)\s*\(\s*{minimum}\s*,",
        expression,
    ) or re.search(
        rf"\b(?:Math\.)?(?:max|Max)\s*\([^,]+,\s*{minimum}\s*\)",
        expression,
    ):
        return True
    if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", compact) is None:
        return False

    containing = tuple(
        definition
        for definition in analyzer.functions
        if definition.start <= index <= definition.end
    )
    start = (
        min(containing, key=lambda item: item.end - item.start).start
        if containing
        else 0
    )
    prefix = analyzer.source[start:index]
    name = re.escape(compact)
    clamp = re.search(
        rf"\b{name}\s*=\s*(?:Math\.)?(?:max|Max)\s*\("
        rf"\s*(?:{name}\s*,\s*{minimum}|{minimum}\s*,\s*{name})\s*\)",
        prefix,
    )
    if clamp is not None:
        return True

    comparisons = (
        rf"{name}\s*<\s*{minimum}",
        rf"{name}\s*<=\s*{minimum - 1}",
        rf"{minimum}\s*>\s*{name}",
        rf"{minimum - 1}\s*>=\s*{name}",
    )
    condition = "(?:" + "|".join(comparisons) + ")"
    return bool(
        re.search(
            rf"\bif\s*\([^)]*{condition}[^)]*\)\s*"
            r"(?:\{[^{}]*(?:throw\b|return\b|process\s*\.\s*exit\s*\()"
            r"|(?:throw\b|return\b|process\s*\.\s*exit\s*\())",
            prefix,
            flags=re.DOTALL,
        )
    )


def _iteration_finding(
    analyzer: SourceAnalyzer,
    call: CallExpression,
    expression: str,
    modern_minimum: int,
) -> DetectorFinding:
    finding = _integer_policy_finding(analyzer, "M07", call, expression, 1000, "kdf_iterations")
    values = _parse_int_values(finding.resolved_values)
    modern_labels = {
        "misuse" if value < modern_minimum else "secure"
        for value in values
    }
    modern_label = "unknown" if not modern_labels else "mixed" if len(modern_labels) > 1 else next(iter(modern_labels))
    return replace(
        finding,
        metadata={
            "paper_minimum_iterations": 1000,
            "paper_policy_label": finding.label,
            "modern_minimum_iterations": modern_minimum,
            "modern_guidance_label": modern_label,
        },
    )


def _key_label(resolved: ResolvedValue) -> tuple[ProgramLabel, str, str]:
    if resolved.origin in {"constant", "predictable", "non_csprng"}:
        return "misuse", "constant_or_predictable_key", f"key origin is {resolved.origin}"
    if resolved.origin in {"secure_random", "derived_secure"}:
        return "secure", "fresh_key_material", f"key origin is {resolved.origin}"
    if resolved.origin == "runtime_input":
        return (
            "unknown",
            "runtime_key_origin_unresolved",
            "runtime-supplied key material is not proven fresh or predictable",
        )
    return "unknown", "unresolved_key_origin", resolved.unresolved_reason or f"key origin is {resolved.origin}"


def _m01_key_label(
    resolved: ResolvedValue,
    *,
    kdf_task: bool,
) -> tuple[ProgramLabel, str, str]:
    if kdf_task and _resolved_value_is_registered_kdf_output(resolved):
        return (
            "secure",
            "registered_kdf_derived_key",
            "key is produced by a registered KDF; KDF arguments are evaluated by M06-M08",
        )
    if _resolved_value_uses_unapproved_password_to_key_transform(resolved):
        return (
            "misuse",
            "predictable_password_to_key_transform",
            "password material is padded, truncated, or hashed directly instead of using an approved KDF",
        )
    return _key_label(resolved)


def _resolved_value_uses_unapproved_password_to_key_transform(
    resolved: ResolvedValue,
) -> bool:
    expressions = (resolved.expression,) + tuple(
        step.expression for step in resolved.steps
    )
    joined = "\n".join(expressions)
    if (
        re.search(
            r"(?i)\b(?:password|passphrase|passwd|pwd|credential|getpass)\b",
            joined,
        )
        is None
        or _resolved_value_is_registered_kdf_output(resolved)
    ):
        return False
    if resolved.origin == "derived_secure" and any(
        step.result == "secure_random"
        and re.search(
            r"\b(?:key|secret|seed|material)\b",
            step.action + " " + step.expression,
            flags=re.IGNORECASE,
        )
        for step in resolved.steps
    ):
        return False
    return bool(
        re.search(
            r"(?i)(?:\.\s*(?:ljust|rjust)\s*\(|"
            r"\[(?::[^]]*|[^]]*:)[^]]*\]|"
            r"(?:SHA(?:256|384|512)|createHash|HashData|ComputeHash)\s*\()",
            joined,
        )
        or any(
            step.action
            in {
                "resolve stateful digest inputs",
                "resolve unapproved password hash transform",
            }
            for step in resolved.steps
        )
    )


def _resolved_value_is_registered_kdf_output(resolved: ResolvedValue) -> bool:
    expressions = (resolved.expression,) + tuple(
        step.expression for step in resolved.steps
    )
    kdf_pattern = re.compile(
        r"(?:"
        r"\b(?:pbkdf2|pbkdf2Sync|pbkdf2_hmac|Pbkdf2|scrypt|scryptSync)\s*\("
        r"|\bPBKDF2HMAC\s*\("
        r"|\bRfc2898DeriveBytes\s*\("
        r"|\b(?:kdf|deriveBytes|deriver)\s*\.\s*GetBytes\s*\("
        r")",
        flags=re.IGNORECASE,
    )
    if any(kdf_pattern.search(expression) for expression in expressions):
        return True
    if any(
        step.action == "resolve registered KDF helper"
        or re.search(
            r"(?i)resolve .*\b(?:Rfc2898DeriveBytes\s*\.\s*Pbkdf2|"
            r"pbkdf2(?:Sync)?|PBKDF2HMAC|scrypt(?:Sync)?)\b",
            step.action,
        )
        for step in resolved.steps
    ):
        return True
    return bool(
        any(
            step.action in {
                "resolve deterministic derivation inputs",
                "resolve custom key-derivation secret inputs",
            }
            for step in resolved.steps
        )
        and any(
            re.search(r"\.\s*(?:derive|GetBytes)\s*\(", item)
            for item in expressions
        )
    )


def _fresh_value_label(resolved: ResolvedValue) -> tuple[ProgramLabel, str, str]:
    if resolved.origin in {"constant", "predictable", "non_csprng"}:
        return "misuse", "constant_or_predictable_value", f"value origin is {resolved.origin}"
    if resolved.origin in {"secure_random", "derived_secure"}:
        return "secure", "fresh_random_value", f"value origin is {resolved.origin}"
    return "unknown", "unresolved_value_origin", resolved.unresolved_reason or f"value origin is {resolved.origin}"


def _csharp_iv_origin(
    analyzer: CStyleAnalyzer,
    expression: str,
    index: int,
) -> ResolvedValue:
    """Treat a digest of stable material as a deterministic CBC IV."""

    resolved = analyzer.resolve_origin(expression, index)
    if resolved.origin != "unknown":
        return resolved
    candidates = [expression.strip(), *(step.expression for step in resolved.steps)]
    bodies = list(candidates)
    for candidate in candidates:
        helper = re.fullmatch(
            r"(?:[A-Za-z_$][A-Za-z0-9_$.]*\s*\.\s*)?"
            r"(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*\([^)]*\)",
            candidate.strip(),
            flags=re.DOTALL,
        )
        if helper is not None:
            bodies.extend(
                analyzer.source[definition.start : definition.end]
                for definition in analyzer.function(helper.group("name"))
            )
    for body in bodies:
        has_digest = re.search(
            r"\b(?:ComputeHash|HashData)\s*\(|"
            r"\b(?:MD5|SHA1|SHA256|SHA384|SHA512)\s*\.\s*Create\s*\(",
            body,
        )
        has_fresh_source = re.search(
            r"\b(?:RandomNumberGenerator|RNGCryptoServiceProvider)\b|"
            r"\bGuid\s*\.\s*NewGuid\s*\(",
            body,
        )
        if has_digest and not has_fresh_source:
            return ResolvedValue(
                expression,
                "predictable",
                unresolved_reason=(
                    "CBC IV is deterministically derived by a local digest helper"
                ),
            )
    return resolved


def _salt_label(resolved: ResolvedValue) -> tuple[ProgramLabel, str, str]:
    expressions = (resolved.expression,) + tuple(
        step.expression for step in resolved.steps
    )
    has_random_wrapper_path = any(
        _is_salt_random_wrapper_call(expression) for expression in expressions
    )
    only_numeric_resolved_values = bool(resolved.values) and all(
        re.fullmatch(r"[-+]?\d+(?:_\d+)*[nNlL]?", value.strip()) is not None
        for value in resolved.values
    )
    only_serialization_metadata_values = bool(resolved.values) and all(
        value.strip()
        in {
            "'='",
            '"="',
            "b'='",
            'b"="',
            "b'\\n'",
            'b"\\n"',
            "'\\n'",
            '"\\n"',
        }
        or re.fullmatch(r"[-+]?\d+(?:_\d+)*[nNlL]?", value.strip())
        is not None
        for value in resolved.values
    )
    has_predictable_fallback = any(
        step.result in {"constant", "predictable", "non_csprng"}
        and step.action
        in {
            "resolve argparse default",
            "resolve assignment",
            "resolve conditional branch",
            "resolve fallback branch",
        }
        for step in resolved.steps
    )
    has_fixed_source_branch = any(
        step.result == "constant"
        and any(
            step.action.startswith(prefix)
            for prefix in (
                "resolve argparse default",
                "resolve assignment salt",
                "resolve assignment kdf_salt",
                "resolve assignment salt_value",
                "resolve assignment saltBytes",
                "resolve caller arguments",
                "resolve conditional branch",
                "resolve fallback branch",
            )
        )
        and re.fullmatch(r"[-+]?\d+(?:_\d+)*[nNlL]?", step.expression.strip())
        is None
        and not _is_salt_output_buffer_allocation(step.expression)
        and not _is_salt_random_wrapper_call(step.expression)
        and not (
            has_random_wrapper_path
            and step.action.startswith("resolve assignment")
            and re.fullmatch(
                r"[A-Za-z_$][A-Za-z0-9_$]*",
                step.expression.strip(),
            )
            is not None
        )
        and not (
            step.action == "resolve caller arguments"
            and re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", step.expression.strip())
            is not None
            and (
                only_numeric_resolved_values
                or only_serialization_metadata_values
                or re.search(r"salt", step.expression, re.IGNORECASE) is None
            )
        )
        for step in resolved.steps
    )
    has_substantive_fixed_value = any(
        value.strip()
        not in {
            "",
            "'='",
            '"="',
            "b'='",
            'b"="',
            "b'\\n'",
            'b"\\n"',
            "'\\n'",
            '"\\n"',
        }
        and re.fullmatch(r"[-+]?\d+(?:_\d+)*[nNlL]?", value.strip()) is None
        and not _is_salt_output_buffer_allocation(value)
        and not _is_salt_random_wrapper_call(value)
        for value in resolved.values
    )
    has_fresh_persisted_origin = _salt_has_fresh_persisted_origin(resolved)
    has_fresh_insecure_runtime_origin = (
        _salt_has_fresh_insecure_runtime_origin(resolved)
    )
    if _salt_resolution_is_password_derived(resolved):
        has_fresh_branch = any(
            step.result == "secure_random"
            or re.search(
                r"\b(?:os\s*\.\s*urandom|secrets\s*\.\s*token_bytes|"
                r"randomBytes|RandomNumberGenerator)\b",
                step.expression,
            )
            is not None
            for step in resolved.steps
        )
        if has_fresh_branch and any(
            step.action.startswith("merge conditional assignment sources")
            for step in resolved.steps
        ):
            return (
                "unknown",
                "runtime_salt_freshness_unresolved",
                "one salt sink selects fresh or password-derived material at runtime",
            )
        if has_fresh_branch:
            return (
                "mixed",
                "mixed_salt_sources",
                "reachable salt sources include fresh random and password-derived material",
            )
        return (
            "misuse",
            "password_derived_salt",
            "salt is deterministically derived from the password material",
        )
    if (
        has_fresh_insecure_runtime_origin
        and not has_fixed_source_branch
        and not has_predictable_fallback
    ):
        return (
            "secure",
            "fresh_runtime_salt",
            "salt varies per operation; randomness quality is evaluated by M05",
        )
    if resolved.origin in {"constant", "predictable", "non_csprng"} and (
        has_fixed_source_branch or has_substantive_fixed_value
    ):
        return (
            "misuse",
            "constant_or_predictable_value",
            "a reachable salt source is constant or predictable",
        )
    if (
        has_fresh_persisted_origin
        and not has_fixed_source_branch
        and not has_predictable_fallback
    ):
        return (
            "secure",
            "fresh_or_persisted_random_salt",
            "salt is freshly generated or reloads the persisted random value",
        )
    if (
        resolved.origin == "secure_random"
        and any(step.result == "secure_random" for step in resolved.steps)
        and not has_fixed_source_branch
        and not has_predictable_fallback
    ):
        return (
            "secure",
            "fresh_runtime_salt",
            "salt retains a reachable fresh-random origin",
        )
    if resolved.origin in {"constant", "predictable", "non_csprng"} or (
        resolved.origin == "runtime_input" and has_predictable_fallback
    ):
        return (
            "misuse",
            "constant_or_predictable_value",
            "a reachable salt source is constant or predictable",
        )
    if any(
        re.search(
            r"\bGuid\s*\.\s*NewGuid\s*\(\s*\)\s*\.\s*ToByteArray\s*\(",
            expression,
        )
        for expression in expressions
    ):
        return (
            "secure",
            "fresh_unique_salt",
            "salt is derived from a newly generated GUID",
        )
    has_runtime_origin = any(
        step.result == "runtime_input" for step in resolved.steps
    )
    if has_runtime_origin:
        if _runtime_salt_is_persisted(resolved):
            return (
                "secure",
                "fresh_or_persisted_random_salt",
                "salt is freshly generated or reloads the persisted random value",
            )
        return (
            "unknown",
            "runtime_salt_freshness_unresolved",
            "caller-provided salt is not proven freshly generated",
        )
    if (
        resolved.origin == "unknown"
        and (
            has_fresh_persisted_origin
            or _salt_has_fresh_extended_origin(resolved)
        )
    ) or (
        resolved.origin == "secure_random"
        and has_fresh_persisted_origin
        and _salt_has_only_persisted_environment_origins(resolved)
    ):
        return (
            "secure",
            "fresh_or_persisted_random_salt",
            "salt is freshly generated or reloads the persisted random value",
        )
    if any(
        re.search(
            r"\b(?:process\s*\.\s*env|Deno\s*\.\s*env|"
            r"Environment\s*\.\s*GetEnvironmentVariable|"
            r"System\s*\.\s*getenv|os\s*\.\s*(?:environ|getenv))\b",
            step.expression,
        )
        for step in resolved.steps
    ):
        # A caller- or environment-supplied salt may be unique, but provenance
        # alone cannot prove freshness. This also covers mixed expressions
        # such as ``configuredSalt ?? randomBytes(16)``: the random fallback
        # does not make the configured branch fresh.
        return (
            "unknown",
            "runtime_salt_freshness_unresolved",
            "runtime-provided salt is not proven freshly generated",
        )
    return _fresh_value_label(resolved)


def _is_salt_output_buffer_allocation(expression: str) -> bool:
    """An empty output buffer is storage, not a fixed salt value."""

    value = expression.strip()
    return bool(
        re.fullmatch(
            r"(?:new\s+byte\s*\[[^\]]+\]|"
            r"Buffer\s*\.\s*alloc(?:Unsafe)?\s*\([^)]*\)|"
            r"bytearray\s*\([^)]*\))",
            value,
            flags=re.IGNORECASE,
        )
    )


def _is_salt_random_wrapper_call(expression: str) -> bool:
    return bool(
        re.match(
            r"\s*(?:drawRandomBytes|DrawRandomBytes|randomBytes|"
            r"secrets\s*\.\s*token_bytes|random\s*\.\s*randbytes)\s*\(",
            expression,
        )
    )


def _salt_has_fresh_insecure_runtime_origin(resolved: ResolvedValue) -> bool:
    expressions = (resolved.expression,) + tuple(
        step.expression for step in resolved.steps
    )
    evidence = "\n".join(expressions)
    fixed_seed = re.search(
        r"(?:\bnew\s+(?:System\.)?Random\s*\(\s*(?:[-+]?\d+|['\"])|"
        r"\brandom\s*\.\s*Random\s*\(\s*(?:[-+]?\d+|['\"]))",
        evidence,
        flags=re.IGNORECASE,
    )
    explicitly_formatted_timestamp = re.search(
        r"\bDateTime\s*\.\s*(?:Now|UtcNow)\s*\.\s*"
        r"ToString\s*\(\s*[^)\s]",
        evidence,
        flags=re.IGNORECASE,
    )
    directly_assigned_ticks = bool(
        re.search(
            r"\bBitConverter\s*\.\s*GetBytes\s*\(",
            evidence,
            flags=re.IGNORECASE,
        )
        and re.search(r"\.\s*Ticks\b", evidence, flags=re.IGNORECASE)
        and not any(
            step.action.startswith("resolve later buffer write")
            for step in resolved.steps
        )
    )
    if (
        fixed_seed is not None
        or explicitly_formatted_timestamp is not None
        or directly_assigned_ticks
    ):
        return False
    return bool(
        re.search(
            r"(?:\bMath\s*\.\s*random\s*\(|"
            r"\bnew\s+(?:System\.)?Random\s*\(\s*\)|"
            r"\b(?:System\s*\.\s*)?Random\s*\.\s*Shared\s*\.\s*NextBytes\s*\(|"
            r"\b[A-Za-z_]\w*\s*\.\s*NextBytes\s*\(|"
            r"\brandom\s*\.\s*(?:random|randbytes|randint|randrange|getrandbits)\s*\(|"
            r"\bDate\s*\.\s*now\s*\(|\bDateTime\s*\.\s*(?:Now|UtcNow)\b|"
            r"\bDateTimeOffset\b|"
            r"\bEnvironment\s*\.\s*TickCount\b|"
            r"\btime\s*\.\s*time(?:_ns)?\s*\()",
            evidence,
            flags=re.IGNORECASE,
        )
    )


def _runtime_salt_is_persisted(resolved: ResolvedValue) -> bool:
    expressions = tuple(step.expression for step in resolved.steps)
    has_fresh_source = any(
        step.result == "secure_random"
        or re.search(
            r"\b(?:randomBytes|randomFillSync|os\s*\.\s*urandom|"
            r"secrets\s*\.\s*token_bytes|RandomNumberGenerator\s*\.\s*GetBytes|"
            r"SecureRandom)\b",
            step.expression,
        )
        for step in resolved.steps
    )
    has_direct_runtime_source = any(
        re.search(
            r"(?:\bargs\s*\[|\bprocess\s*\.\s*argv\s*\[|"
            r"\bsys\s*\.\s*argv\s*\[|\bprocess\s*\.\s*env\b|"
            r"\bDeno\s*\.\s*env\b|"
            r"\bEnvironment\s*\.\s*GetEnvironmentVariable\b|"
            r"\bSystem\s*\.\s*getenv\b|"
            r"\bos\s*\.\s*(?:environ|getenv)\b|"
            r"\b(?:input|prompt|question)\s*\(|"
            r"\bConsole\s*\.\s*ReadLine\s*\()",
            expression,
        )
        for expression in expressions
    )
    has_explicit_persisted_context = any(
        re.search(
            r"(?i)\b(?:stored|existing|record|packet|payload|envelope|serialized|verify|"
            r"file|path|loaded|replay|token|parsed|decoded)\b",
            expression,
        )
        or re.search(
            r"(?i)(?:^|_)(?:stored|existing|record|packet|payload|envelope|serialized|"
            r"verify|file|path|loaded|replay|token|parsed|decoded)(?:_|$)",
            expression,
        )
        or _looks_like_serialized_salt_prompt(expression)
        or re.search(
            r"(?i)(?:file read|BlockCopy|later buffer write)",
            step.action,
        )
        for expression, step in zip(expressions, resolved.steps)
    ) or any(_looks_like_persisted_salt_expression(item) for item in expressions)
    if has_direct_runtime_source:
        # Decoding an arbitrary CLI/environment value as hex or Base64 is not
        # evidence that it reloads a salt generated by this application. A
        # direct runtime source needs explicit record/file/payload context.
        return has_explicit_persisted_context or (
            has_fresh_source
            and any(
                _looks_like_serialized_salt_question(item)
                for item in expressions
            )
        )
    return (
        has_explicit_persisted_context
        or any(_looks_like_persisted_salt_expression(item) for item in expressions)
    ) and any(
        _looks_like_persisted_salt_expression(step.expression)
        or re.search(
            r"(?i)\b(?:stored|existing|record|packet|payload|envelope|serialized|verify|"
            r"file|path|loaded|replay|token|parsed|decoded)\b",
            step.expression,
        )
        or re.search(
            r"(?i)(?:^|_)(?:stored|existing|record|packet|payload|envelope|serialized|"
            r"verify|file|path|loaded|replay|token|parsed|decoded)(?:_|$)",
            step.expression,
        )
        or re.search(
            r"(?i)(?:file read|BlockCopy|later buffer write)",
            step.action,
        )
        for step in resolved.steps
    )


def _looks_like_serialized_salt_prompt(expression: str) -> bool:
    return bool(
        re.search(
            r"\b(?:input|prompt|[A-Za-z_$][A-Za-z0-9_$]*Question)\s*\(",
            expression,
            re.IGNORECASE,
        )
        and re.search(r"(?i)\bsalt", expression)
        and re.search(
            r"(?i)(?:base64(?:url)?|b64|hex|stored|saved|token|ciphertext|encrypted)",
            expression,
        )
    )


def _looks_like_serialized_salt_question(expression: str) -> bool:
    return bool(
        re.search(
            r"\b(?:input|prompt|question|[A-Za-z_$][A-Za-z0-9_$]*Question)\s*\(",
            expression,
            re.IGNORECASE,
        )
        and re.search(r"(?i)\bsalt", expression)
        and re.search(r"(?i)(?:base64(?:url)?|b64|hex|stored|saved)", expression)
    )


def _salt_has_only_persisted_environment_origins(
    resolved: ResolvedValue,
) -> bool:
    environment_expressions = tuple(
        step.expression
        for step in resolved.steps
        if re.search(
            r"\b(?:process\s*\.\s*env|Deno\s*\.\s*env|"
            r"Environment\s*\.\s*GetEnvironmentVariable|"
            r"System\s*\.\s*getenv|os\s*\.\s*(?:environ|getenv))\b",
            step.expression,
        )
    )
    if not environment_expressions:
        return False
    return all(
        re.search(
            r"(?:stored|record|payload|envelope|serialized|token|verify|file|path)",
            expression,
            flags=re.IGNORECASE,
        )
        for expression in environment_expressions
    )


def _salt_has_fresh_persisted_origin(resolved: ResolvedValue) -> bool:
    expressions = tuple(step.expression for step in resolved.steps)
    has_fresh_source = any(
        step.result == "secure_random"
        or re.search(
            r"\b(?:randomBytes|randomFillSync|os\s*\.\s*urandom|"
            r"secrets\s*\.\s*token_bytes|RandomNumberGenerator\s*\.\s*GetBytes|"
            r"SecureRandom)\b",
            step.expression,
        )
        for step in resolved.steps
    )
    has_persisted_source = any(
        _looks_like_persisted_salt_expression(expression)
        for expression in expressions
    ) or any(
        re.search(
            r"BlockCopy|later buffer write|file read|persisted file|file value|decode",
            step.action,
            re.IGNORECASE,
        )
        for step in resolved.steps
    )
    return has_fresh_source and has_persisted_source


def _looks_like_persisted_salt_expression(expression: str) -> bool:
    explicit_storage = re.search(
        r"\b(?:readFileSync|readFile|ReadAllBytes|ReadAllText|"
        r"[A-Za-z_$][A-Za-z0-9_$]*\s*\.\s*(?:read|read_bytes|read_text)\s*\(|"
        r"json\s*\.\s*load|JSON\s*\.\s*parse|"
        r"Buffer\s*\.\s*BlockCopy|"
        r"(?:base64\s*\.\s*)?(?:urlsafe_)?b64encode\s*\(|"
        r"[A-Za-z_$][A-Za-z0-9_$]*\s*\.\s*toString\s*\(\s*[\"']base64(?:url)?[\"']|"
        r"[A-Za-z_$][A-Za-z0-9_$]*\s*\.\s*hex\s*\(|"
        r"(?:payload|packet|package|record|envelope|serialized|decoded)"
        r"\s*\.\s*(?:subarray|slice|AsSpan)\s*\(|"
        r"(?:parsed|decoded|existing|envelope|record|rec|fields|msg|blob|payload|loaded|encrypted|replay|"
        r"output[A-Za-z0-9_]*|package[A-Za-z0-9_]*|env|ch)"
        r"(?:\s*\.\s*[A-Za-z_$][A-Za-z0-9_$]*)*"
        r"\s*\.\s*(?:salt|s|saltB64|salt_b64u|salt_b64|salt_b64url|Salt)|"
        r"[A-Za-z_$][A-Za-z0-9_$]*\s*\[\s*[\"']"
        r"(?:salt|saltB64|salt_b64u|salt_b64|salt_b64url|kdf)"
        r"[\"']\s*\]|"
        r"(?:parts|fields|segments)\s*\[\s*\d+\s*\]|"
        r"(?:packaged|combined|sealed|encoded_salt|stored_salt))",
        expression,
        flags=re.IGNORECASE,
    )
    if explicit_storage is not None:
        return True

    # An encoding conversion alone does not establish persistence: CLI and
    # environment overrides are commonly decoded from hex/Base64 too. Accept
    # a decoder as persisted evidence only when its data expression names an
    # actual stored record/container.
    has_decoder = re.search(
        r"\b(?:Buffer\s*\.\s*from|urlsafe_b64decode|b64decode|"
        r"FromBase64String|b64d|fromhex)\s*\(",
        expression,
        flags=re.IGNORECASE,
    )
    has_storage_context = re.search(
        r"\b(?:stored|saved|record|packet|payload|envelope|serialized|"
        r"persisted|loaded|replay|file|path)[A-Za-z0-9_]*\b",
        expression,
        flags=re.IGNORECASE,
    )
    return has_decoder is not None and has_storage_context is not None


def _salt_has_fresh_extended_origin(resolved: ResolvedValue) -> bool:
    has_fresh_source = any(
        step.result == "secure_random"
        or re.search(
            r"\b(?:randomBytes|os\s*\.\s*urandom|"
            r"secrets\s*\.\s*token_bytes|RandomNumberGenerator\s*\.\s*GetBytes)\b",
            step.expression,
        )
        for step in resolved.steps
    )
    has_salt_extension = any(
        step.action == "combine expression"
        and re.search(r"\bsalt\b\s*\+|\+\s*\bsalt\b", step.expression)
        for step in resolved.steps
    )
    return has_fresh_source and has_salt_extension


def _random_label(resolved: ResolvedValue) -> tuple[ProgramLabel, str, str]:
    alternative_origins = {
        step.result
        for step in resolved.steps
        if step.action == "resolve selected callable alternative"
    }
    if alternative_origins & {"secure_random", "derived_secure"} and (
        alternative_origins & {"non_csprng", "predictable", "constant"}
    ):
        return (
            "mixed",
            "mixed_randomness_sources",
            "reachable output generators include both approved and non-cryptographic randomness",
        )
    random_path_origins = {
        step.result
        for step in resolved.steps
        if step.result
        in {"secure_random", "derived_secure", "non_csprng", "predictable"}
    }
    merges_distinct_uses = any(
        step.action
        in {
            "resolve caller arguments",
            "resolve constructed instance attribute",
        }
        for step in resolved.steps
    )
    if merges_distinct_uses and random_path_origins & {
        "secure_random",
        "derived_secure",
    } and random_path_origins & {"non_csprng", "predictable"}:
        return (
            "mixed",
            "mixed_randomness_sources",
            "reachable uses include both approved and non-cryptographic randomness",
        )
    expressions = (resolved.expression,) + tuple(
        step.expression for step in resolved.steps
    )
    resolves_local_random_int = any(
        step.action == "resolve helper randomInt" for step in resolved.steps
    )
    has_secure_source = any(
        re.search(
            r"(?:\b(?:randomBytes|randomFillSync|getRandomValues)\s*\("
            r"|\bRandomNumberGenerator\s*\.\s*(?:GetBytes|Fill|GetInt32)\s*\("
            r"|\bnew\s+(?:java\.security\.)?SecureRandom\s*\("
            r"|\bSecureRandom\s*\.\s*getInstance\s*\("
            r"|\b(?:secrets\s*\.\s*(?:token_bytes|token_hex|token_urlsafe|"
            r"choice|randbits|randbelow)"
            r"|os\s*\.\s*urandom)\s*\()",
            expression,
            flags=re.IGNORECASE,
        )
        for expression in expressions
    )
    if not resolves_local_random_int:
        has_secure_source = has_secure_source or any(
            re.search(r"\brandomInt\s*\(", expression) is not None
            for expression in expressions
        )
    has_weak_source = any(
        re.search(
            r"(?:\bMath\s*\.\s*(?:random|imul)\s*\("
            r"|\bnew\s+(?:System\s*\.)?Random\s*\("
            r"|\b(?:System\s*\.)?Random\s*\.\s*Shared\b"
            r"|\bnew\s+(?:java\.util\.)?Random\s*\("
            r"|\brandom\s*\.\s*(?:Random|random|randbytes|randint|randrange|getrandbits)\s*\("
            r"|\b(?:Date\s*\.\s*now|System\s*\.\s*(?:currentTimeMillis|nanoTime)"
            r"|time\s*\.\s*time(?:_ns)?)\s*\()",
            expression,
            flags=re.IGNORECASE,
        )
        for expression in expressions
    )
    has_alternative_path = bool(alternative_origins) or any(
        "conditional" in step.action
        or "fallback value sources" in step.action
        for step in resolved.steps
    )
    if has_secure_source and has_weak_source and has_alternative_path:
        return (
            "mixed",
            "mixed_randomness_sources",
            "reachable output generators include both approved and non-cryptographic randomness",
        )
    if has_secure_source and any(
        step.action == "resolve template interpolation origins"
        for step in resolved.steps
    ):
        return (
            "secure",
            "approved_csprng",
            "the composed output retains an approved cryptographic random component",
        )
    if resolved.origin in {"non_csprng", "predictable", "constant"}:
        return "misuse", "non_csprng_or_predictable_seed", f"security randomness origin is {resolved.origin}"
    if resolved.origin in {"secure_random", "derived_secure"}:
        return "secure", "approved_csprng", f"security randomness origin is {resolved.origin}"
    return "unknown", "unresolved_randomness_origin", resolved.unresolved_reason or f"randomness origin is {resolved.origin}"


def _password_label(resolved: ResolvedValue) -> tuple[ProgramLabel, str, str]:
    predictable = bool(
        resolved.origin in {"constant", "predictable", "non_csprng"}
        or _password_resolution_has_predictable_fallback(resolved)
    )
    if predictable and _password_resolution_has_secure_random_alternative(
        resolved
    ):
        return (
            "mixed",
            "mixed_password_sources",
            "reachable password sources include both predictable and freshly generated material",
        )
    if predictable and _password_resolution_has_runtime_alternative(resolved):
        return (
            "mixed",
            "mixed_password_sources",
            "reachable password sources include both runtime and predictable material",
        )
    if predictable:
        return (
            "misuse",
            "hardcoded_password",
            "a reachable password source is constant or predictable",
        )
    if resolved.origin == "runtime_input":
        return "secure", "runtime_password_source", "password comes from an approved runtime source"
    if (
        resolved.origin == "unknown"
        and (
            any(step.result == "runtime_input" for step in resolved.steps)
            or any(
                re.search(
                    r"\b(?:process\s*\.\s*(?:env|argv|stdin)|stdin\b|"
                    r"readFileSync\s*\(|Console\s*\.\s*Read|"
                    r"Environment\s*\.\s*GetEnvironmentVariable|"
                    r"getpass\s*\.\s*getpass\s*\(|"
                    r"[A-Za-z_][A-Za-z0-9_]*\s*\.\s*read\s*\()",
                    step.expression,
                )
                for step in resolved.steps
            )
        )
        and not _password_resolution_has_predictable_fallback(resolved)
    ):
        return (
            "secure",
            "guarded_runtime_password_source",
            "runtime password alternatives remain after empty sentinels are rejected",
        )
    if resolved.origin in {"secure_random", "derived_secure"}:
        return (
            "secure",
            "fresh_password_material",
            f"password material origin is {resolved.origin}",
        )
    return "unknown", "unresolved_password_origin", resolved.unresolved_reason or f"password origin is {resolved.origin}"


def _password_resolution_has_secure_random_alternative(
    resolved: ResolvedValue,
) -> bool:
    if _resolved_value_is_registered_kdf_output(resolved):
        # When a KDF output is itself used as a password, fresh salt or a
        # generic "derived_secure" marker belongs to the derivation.  It is
        # not an alternative password source and must not turn a fixed KDF
        # password into mixed M08 evidence.
        return False
    return any(
        step.result in {"secure_random", "derived_secure"}
        or re.search(
            r"\b(?:secrets\s*\.\s*(?:token_bytes|token_hex|token_urlsafe|choice)|"
            r"os\s*\.\s*urandom|randomBytes|RandomNumberGenerator\s*\.\s*GetBytes)\b",
            step.expression,
        )
        is not None
        for step in resolved.steps
    )


def _password_resolution_has_predictable_fallback(
    resolved: ResolvedValue,
) -> bool:
    """Recognize fixed or public-host-derived password fallback branches."""

    candidates = tuple(
        step.expression
        for step in resolved.steps
        if step.result in {"constant", "predictable"}
        and any(
            marker in step.action
            for marker in ("fallback", "default", "conditional branch")
        )
    )
    for expression in candidates:
        value = expression.strip()
        if value.startswith("`"):
            static_text = re.sub(r"\$\{[^{}]*\}", "", value)
        elif value.startswith('$"'):
            static_text = re.sub(r"\{[^{}]*\}", "", value)
        else:
            continue
        if re.search(
            r"(?i)(?:fallback|default|demo|local|password|passphrase)",
            static_text,
        ):
            return True
    return False


def _classify_ts_mode(value: str) -> tuple[ProgramLabel, str, str]:
    normalized = value.lower().replace("_", "-")
    if normalized in {"aes128", "aes192", "aes256"}:
        return (
            "secure",
            "approved_explicit_mode",
            f"cipher algorithm {value!r} is a Node/OpenSSL AES-CBC alias",
        )
    if "ecb" in normalized:
        return "misuse", "insecure_ecb_mode", f"cipher algorithm {value!r} selects ECB"
    if any(mode in normalized for mode in ("cbc", "gcm", "ccm", "ctr", "cfb", "ofb", "chacha20-poly1305")):
        return "secure", "approved_explicit_mode", f"cipher algorithm {value!r} selects an approved explicit mode"
    return "unknown", "unclassified_cipher_mode", f"cipher algorithm {value!r} has no classified mode"


def _with_typescript_cipher_mode_fragments(
    resolved: ResolvedValue,
) -> ResolvedValue:
    mode_pattern = re.compile(
        r"(?:^|[-_])(?P<mode>ecb|cbc|gcm|ccm|ctr|cfb|ofb)(?:$|[-_])",
        flags=re.IGNORECASE,
    )
    if resolved.values and all(
        mode_pattern.search(value) for value in resolved.values
    ):
        return resolved
    expressions = (resolved.expression,) + tuple(
        step.expression for step in resolved.steps
    )
    modes: set[str] = set()
    for expression in expressions:
        modes.update(
            match.group("mode").lower()
            for match in re.finditer(
                r"(?:\+\s*)?[\"'`](?:-)?"
                r"(?P<mode>ecb|cbc|gcm|ccm|ctr|cfb|ofb)[\"'`]"
                r"(?:\s*\+)?",
                expression,
                flags=re.IGNORECASE,
            )
            if "+" in match.group(0) or "${" in expression
        )
    if not modes:
        return resolved
    return replace(
        resolved,
        origin="constant",
        values=tuple(f"aes-unknown-{mode}" for mode in sorted(modes)),
        unresolved_reason=None,
    )


def _classify_csharp_mode(value: str) -> tuple[ProgramLabel, str, str]:
    normalized = _normalize_algorithm(value)
    if normalized == "ECB":
        return "misuse", "insecure_ecb_mode", "CipherMode.ECB is insecure"
    if normalized in {"CBC", "CFB", "OFB", "CTS"}:
        return "secure", "approved_explicit_mode", f"CipherMode.{value} is approved under the benchmark mode policy"
    return "unknown", "unclassified_cipher_mode", f"CipherMode {value!r} is not classified"


def _classify_python_mode(value: str) -> tuple[ProgramLabel, str, str]:
    normalized = _normalize_algorithm(value)
    if normalized == "ECB":
        return "misuse", "insecure_ecb_mode", "modes.ECB is insecure"
    if normalized in {"CBC", "GCM", "CTR", "CFB", "CFB8", "OFB", "XTS"}:
        return (
            "secure",
            "approved_explicit_mode",
            f"modes.{value} is approved under the benchmark mode policy",
        )
    return (
        "unknown",
        "unclassified_cipher_mode",
        f"Python cipher mode {value!r} is not classified",
    )


def _python_mode_constructor_target(expression: str) -> str | None:
    try:
        node = ast.parse(expression.strip(), mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(node, ast.Call):
        return None
    if not isinstance(node.func, (ast.Name, ast.Attribute)):
        return None
    return ast.unparse(node.func)


def _classify_digest(value: str) -> tuple[ProgramLabel, str, str]:
    normalized = _normalize_algorithm(value)
    if normalized in _BROKEN_DIGESTS:
        return "misuse", "broken_digest", f"digest {value!r} is broken or deprecated"
    if normalized in _SECURE_DIGESTS:
        return "secure", "approved_digest", f"digest {value!r} is approved"
    return "unknown", "unclassified_digest", f"digest {value!r} is not classified"


def _classify_mac(value: str) -> tuple[ProgramLabel, str, str]:
    normalized = _normalize_algorithm(value)
    digest_name = normalized in (_SECURE_DIGESTS | _BROKEN_DIGESTS)
    if (
        not normalized.startswith(("HMAC", "PBEWITHHMAC", "KMAC"))
        and normalized != "AESCMAC"
        and not digest_name
    ):
        return "unknown", "unclassified_mac", f"MAC algorithm {value!r} is not classified"
    if digest_name:
        normalized = "HMAC" + normalized
    if normalized in _BROKEN_MACS:
        return "misuse", "compromised_mac", f"MAC algorithm {value!r} is broken or deprecated"
    if normalized in _SECURE_MACS:
        return "secure", "approved_mac", f"MAC algorithm {value!r} is approved"
    return "unknown", "unclassified_mac", f"MAC algorithm {value!r} is not classified"


def _result(
    analyzer: SourceAnalyzer,
    misuse_id: str,
    findings: Iterable[DetectorFinding],
    no_sink_reason: str,
) -> DetectionResult:
    normalized = [
        finding
        if finding.policy_id
        else replace(finding, policy_id=f"{misuse_id}.policy.v1")
        for finding in findings
    ]
    return combine_detector_findings(
        misuse_id,
        analyzer.language,
        _dedupe_findings(normalized),
        no_sink_reason=no_sink_reason,
    )


def _dedupe_findings(findings: Iterable[DetectorFinding]) -> list[DetectorFinding]:
    records: dict[tuple[str, int, str, ProgramLabel], DetectorFinding] = {}
    for finding in findings:
        key = (finding.sink_id, finding.span.start, finding.expression, finding.label)
        records.setdefault(key, finding)
    return sorted(records.values(), key=lambda item: (item.span.start, item.sink_id, item.expression))


def _suppress_unresolved_internal_adapters(
    findings: Iterable[DetectorFinding],
    *,
    adapter_names: set[str],
) -> list[DetectorFinding]:
    """Drop only redundant unresolved native-adapter observations.

    Some translated programs preserve a task-facing Java-shaped helper and
    implement it with a target-native MAC call. The controlled algorithm is at
    the outer helper call; an inner adapter parameter can be intentionally
    opaque to the bounded resolver. Once a non-adapter sink has a resolved
    policy label, that unresolved implementation detail must not downgrade the
    program to unknown. A sole or resolved adapter finding is retained.
    """

    records = list(findings)
    has_resolved_outer_sink = any(
        finding.label != "unknown"
        and finding.sink_name.rsplit(".", 1)[-1] not in adapter_names
        for finding in records
    )
    if not has_resolved_outer_sink:
        return records
    return [
        finding
        for finding in records
        if not (
            finding.label == "unknown"
            and finding.sink_name.rsplit(".", 1)[-1] in adapter_names
        )
    ]


def _suppress_unresolved_uncalled_helpers(
    analyzer: SourceAnalyzer,
    findings: Iterable[DetectorFinding],
) -> list[DetectorFinding]:
    """Ignore an unresolved sink in a helper that the program never calls."""

    records = list(findings)
    if not any(finding.label != "unknown" for finding in records):
        return records
    kept: list[DetectorFinding] = []
    for finding in records:
        if finding.label != "unknown":
            kept.append(finding)
            continue
        function_name = analyzer.containing_function_name(finding.span.start)
        if function_name is None or analyzer.containing_function_is_called(
            finding.span.start
        ):
            kept.append(finding)
    return kept


def _python_m01_internal_kdf_hmac_finding(
    analyzer: SourceAnalyzer,
    finding: DetectorFinding,
) -> bool:
    """Exclude HMAC invocations that implement a project-local KDF.

    RFC 5869-style helpers legitimately use an all-zero extract key when no
    salt is supplied and initialize their output accumulator with ``b""``.
    Those implementation details are not application cryptographic keys.  M01
    still evaluates every AES or application-HMAC key returned by the helper.
    """

    if not isinstance(analyzer, PythonAnalyzer):
        return False
    if "hmac" not in finding.sink_name.lower():
        return False
    function_name = analyzer.containing_function_name(finding.span.start)
    if function_name is None:
        return False
    return re.search(
        r"(?i)(?:derive|derivation|hkdf|kdf|expand).*(?:key|token|secret|material)"
        r"|(?:key|token|secret|material).*(?:derive|derivation|hkdf|kdf|expand)"
        r"|(?:^|_)(?:hkdf|kdf)(?:_|$)",
        function_name,
    ) is not None


def _suppress_custom_prng_seed_findings(
    analyzer: SourceAnalyzer,
    findings: Iterable[DetectorFinding],
) -> list[DetectorFinding]:
    """A CSPRNG seed does not make a custom LCG output a secure PRNG."""

    records = list(findings)
    lcg_positions = [
        finding.span.start
        for finding in records
        if finding.label == "misuse" and finding.sink_name == "Math.imul"
    ]
    if not lcg_positions:
        return records

    retained: list[DetectorFinding] = []
    for finding in records:
        if finding.label != "secure" or finding.span.start > max(lcg_positions):
            retained.append(finding)
            continue
        statement_start = max(
            analyzer.source.rfind(";", 0, finding.span.start),
            analyzer.source.rfind("\n", 0, finding.span.start),
            analyzer.source.rfind("{", 0, finding.span.start),
        ) + 1
        statement_end = analyzer.source.find(";", finding.span.end)
        if statement_end == -1:
            retained.append(finding)
            continue
        statement = analyzer.source[statement_start : statement_end + 1]
        assigned = re.search(
            r"\b(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)"
            r"(?:\s*:[^=;]+)?\s*=",
            statement,
        )
        if assigned is None:
            retained.append(finding)
            continue
        state_name = assigned.group(1)
        later = analyzer.source[statement_end + 1 :]
        if not re.search(
            rf"\bMath\s*\.\s*imul\s*\(\s*{re.escape(state_name)}\b",
            later,
        ):
            retained.append(finding)
    return retained


def _with_operation_direction(
    analyzer: SourceAnalyzer,
    finding: DetectorFinding,
    index: int,
    operation_name: str,
) -> DetectorFinding:
    normalized = operation_name.lower()
    if isinstance(analyzer, CStyleAnalyzer) and analyzer.language == "typescript":
        normalized = next(
            (
                imported.lower()
                for imported, alias in analyzer.import_aliases.items()
                if alias.lower() == normalized
            ),
            normalized,
        )
    function_name = (analyzer.containing_function_name(index) or "").lower()
    operation_has_encrypt = "encrypt" in normalized or "encipher" in normalized
    operation_has_decrypt = "decrypt" in normalized or "decipher" in normalized
    function_has_encrypt = "encrypt" in function_name or "encipher" in function_name
    function_has_decrypt = "decrypt" in function_name or "decipher" in function_name
    if operation_has_encrypt and operation_has_decrypt:
        direction = "encrypt"
    elif operation_has_decrypt:
        direction = "decrypt"
    elif (
        "cipheriv" in normalized
        and function_has_decrypt
        and not function_has_encrypt
    ):
        # A compatibility wrapper may call createCipheriv inside a
        # decrypt-named helper. The containing operation still determines that
        # this IV was transported with ciphertext rather than chosen for
        # encryption.
        direction = "decrypt"
    elif operation_has_encrypt or "cipheriv" in normalized:
        direction = "encrypt"
    elif function_has_encrypt and function_has_decrypt:
        # A combined round-trip helper chooses the IV at encryption time and
        # reuses it for decryption. Keep that one security decision visible.
        direction = "encrypt"
    elif function_has_decrypt:
        direction = "decrypt"
    elif function_has_encrypt:
        direction = "encrypt"
    elif function_name:
        local_direction = _operation_direction_from_local_context(analyzer, index)
        if local_direction != "setup":
            # The terminal operation inside the helper is stronger evidence
            # than a caller name.  In particular, decryption fixtures often
            # call an encryption helper to construct compatible ciphertext.
            direction = local_direction
        else:
            caller_directions = _helper_caller_operation_directions(
                analyzer,
                function_name,
                index,
            )
            if "encrypt" in caller_directions:
                direction = "encrypt"
            elif caller_directions == {"decrypt"}:
                direction = "decrypt"
            else:
                direction = "setup"
    else:
        direction = _operation_direction_from_local_context(analyzer, index)
    return replace(
        finding,
        metadata={**finding.metadata, "operation_direction": direction},
    )


def _helper_caller_operation_directions(
    analyzer: SourceAnalyzer,
    function_name: str,
    definition_index: int,
) -> set[str]:
    directions: set[str] = set()
    for match in re.finditer(
        rf"\b{re.escape(function_name)}\s*\(",
        analyzer.source,
    ):
        caller = (analyzer.containing_function_name(match.start()) or "").lower()
        if caller == function_name:
            continue
        if "encrypt" in caller or "encipher" in caller:
            directions.add("encrypt")
        elif "decrypt" in caller or "decipher" in caller:
            directions.add("decrypt")
    return directions


def _operation_direction_from_local_context(
    analyzer: SourceAnalyzer,
    index: int,
) -> str:
    context = analyzer.source[
        max(0, index - 40) : min(len(analyzer.source), index + 240)
    ].lower()
    has_decrypt = "decrypt" in context or "decipher" in context
    has_encrypt = "encrypt" in context or "encipher" in context
    if has_encrypt:
        # A value prepared once and then passed to both operations is the
        # encryption-time IV choice, even when a later decrypt call is in the
        # same local context window.
        return "encrypt"
    if has_decrypt:
        return "decrypt"
    return "setup"


def _suppress_unresolved_decryption_reuse(
    findings: Iterable[DetectorFinding],
) -> list[DetectorFinding]:
    """Evaluate IV freshness where encryption chooses the IV.

    A decryptor must consume the IV transported with the ciphertext. A literal
    fixture at that sink does not prove that an encryptor generated a
    predictable IV, and randomizing it would only make the fixture unusable.
    """

    records = list(findings)
    return [
        finding
        for finding in records
        if finding.metadata.get("operation_direction") != "decrypt"
    ]


def _suppress_proven_inactive_findings(
    analyzer: SourceAnalyzer,
    findings: Iterable[DetectorFinding],
) -> list[DetectorFinding]:
    """Drop a finding only when constant local control flow proves it dormant."""

    return [
        finding
        for finding in findings
        if finding.span.start < 0
        or analyzer.position_is_reachable(finding.span.start)
    ]


def _preserve_definite_inactive_findings(
    analyzer: SourceAnalyzer,
    lexical_findings: Iterable[DetectorFinding],
    active_findings: Iterable[DetectorFinding],
) -> list[DetectorFinding]:
    """Keep definite insecure APIs visible for source-cleaning purposes.

    Reachability remains useful for dropping unresolved or harmless dormant
    branches.  A concrete weak primitive or fixed security value, however,
    remains in the model input and must be normalized before representation
    extraction even when one constant demo call currently avoids that branch.
    """

    records = list(active_findings)
    if not getattr(analyzer, "include_inactive_findings", False):
        return records
    signatures = {
        (item.span.start, item.span.end, item.expression, item.label)
        for item in records
    }
    records.extend(
        item
        for item in lexical_findings
        if item.label in {"misuse", "mixed"}
        and (item.span.start, item.span.end, item.expression, item.label)
        not in signatures
    )
    return records


def _suppress_unresolved_verification_reuse(
    analyzer: SourceAnalyzer,
    findings: Iterable[DetectorFinding],
) -> list[DetectorFinding]:
    """Ignore an unresolved persisted salt at a verification-only KDF sink.

    A program may generate and store a fresh salt while creating a password
    record, then read the stored salt during verification. Once the creation
    sink has a resolved origin, an opaque ``verify``/``check`` read is reuse of
    that value rather than a second salt-generation decision.
    """

    records = list(findings)
    if not any(finding.label != "unknown" for finding in records):
        return records
    verification_terms = ("verif", "check", "authenticate", "compare", "match")
    has_resolved_origin = any(finding.label != "unknown" for finding in records)
    return [
        finding
        for finding in records
        if not (
            finding.label == "unknown"
            and (
                any(
                    term in (analyzer.containing_function_name(finding.span.start) or "").lower()
                    for term in verification_terms
                )
                or (
                    has_resolved_origin
                    and _finding_looks_like_persisted_salt_reuse(finding)
                )
            )
        )
    ]


def _finding_looks_like_persisted_salt_reuse(
    finding: DetectorFinding,
) -> bool:
    expressions = (finding.expression,) + tuple(
        step.expression for step in finding.resolution_steps
    )
    return any(
        _looks_like_persisted_salt_expression(expression)
        for expression in expressions
    ) or any(
        _looks_like_serialized_salt_prompt(expression)
        for expression in expressions
    ) or any(
        re.search(r"BlockCopy|later buffer write|file read|decode", step.action, re.IGNORECASE)
        for step in finding.resolution_steps
    )


def _missing_argument_finding(
    analyzer: SourceAnalyzer,
    misuse_id: str,
    call: CallExpression,
    sink_id: str,
) -> DetectorFinding:
    return DetectorFinding(
        misuse_id=misuse_id,
        language=analyzer.language,
        sink_id=sink_id,
        sink_name=call.callee,
        expression="",
        label="unknown",
        reason_code="target_argument_missing",
        reason="required security-policy argument is missing",
        span=analyzer.source_span(call.start, call.end),
    )


def _direct_call_finding(
    analyzer: SourceAnalyzer,
    misuse_id: str,
    call: CallExpression,
    label: ProgramLabel,
    code: str,
    reason: str,
) -> DetectorFinding:
    return DetectorFinding(
        misuse_id=misuse_id,
        language=analyzer.language,
        sink_id=f"{misuse_id.lower()}_api",
        sink_name=call.callee,
        expression=analyzer.source[call.start : call.end],
        label=label,
        reason_code=code,
        reason=reason,
        span=analyzer.source_span(call.start, call.end),
    )


def _direct_policy_finding(
    analyzer: SourceAnalyzer,
    misuse_id: str,
    sink_id: str,
    expression: str,
    label: ProgramLabel,
    code: str,
    reason: str,
) -> DetectorFinding:
    match = re.search(re.escape(expression.split()[0]), analyzer.source)
    start = match.start() if match else -1
    end = match.end() if match else -1
    return DetectorFinding(
        misuse_id=misuse_id,
        language=analyzer.language,
        sink_id=sink_id,
        sink_name=expression,
        expression=expression,
        label=label,
        reason_code=code,
        reason=reason,
        span=analyzer.source_span(start, end) if start >= 0 else SourceSpan(),
    )


def _object_property(expression: str, name: str) -> str | None:
    value = expression.strip()
    if not (value.startswith("{") and value.endswith("}")):
        return None
    body = value[1:-1]
    synthetic = f"({body})"
    for item in split_arguments(synthetic, 0, len(synthetic) - 1):
        candidate = item.strip()
        if candidate == name:
            return name
        match = re.fullmatch(
            rf"(?:[\"']?{re.escape(name)}[\"']?)\s*:\s*(?P<value>.+)",
            candidate,
            flags=re.DOTALL,
        )
        if match is not None:
            return match.group("value").strip()
    return None


def _csharp_object_initializer_property(
    expression: str,
    name: str,
) -> str | None:
    value = expression.strip()
    opening = value.find("{")
    if opening == -1 or not value.endswith("}"):
        return None
    for item in split_arguments(value, opening, len(value) - 1):
        match = re.fullmatch(
            rf"{re.escape(name)}\s*=\s*(?P<value>.+)",
            item.strip(),
            flags=re.DOTALL,
        )
        if match is not None:
            return match.group("value").strip()
    return None


def _call_argument(
    call: CallExpression,
    index: int,
    *names: str,
) -> str | None:
    for name in names:
        if name in call.named_arguments:
            return call.named_arguments[name]
    if len(call.arguments) <= index:
        return None
    return _argument_value(call.arguments[index])


def _argument_value(expression: str) -> str:
    expression = expression.strip()
    walrus_expression = (
        expression[1:-1].strip()
        if expression.startswith("(") and expression.endswith(")")
        else expression
    )
    walrus = re.fullmatch(
        r"[A-Za-z_]\w*\s*:=\s*(?P<value>.+)",
        walrus_expression,
        flags=re.DOTALL,
    )
    if walrus is not None:
        return walrus.group("value").strip()
    expression = re.sub(r"^=+\s*", "", expression)
    match = re.match(
        r"^[A-Za-z_$][A-Za-z0-9_$]*\s*:\s*(?P<value>.+)$",
        expression,
        flags=re.DOTALL,
    )
    expression = match.group("value").strip() if match else expression
    return expression


def _first_call_argument(expression: str) -> str | None:
    opening = expression.find("(")
    closing = expression.rfind(")")
    if opening == -1 or closing <= opening:
        return None
    return expression[opening + 1 : closing].split(",", 1)[0].strip()


def _parse_int_values(values: Iterable[str]) -> set[int]:
    parsed: set[int] = set()
    for value in values:
        try:
            parsed.add(int(str(value).replace("_", ""), 10))
        except ValueError:
            continue
    return parsed

def _normalize_algorithm(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", value).upper()


def _program_label(value: object) -> ProgramLabel:
    text = str(value)
    return text if text in {"secure", "misuse", "mixed", "unknown"} else "unknown"  # type: ignore[return-value]


def _reason_code_for_label(label: ProgramLabel) -> str:
    return {
        "secure": "java_secure_sink",
        "misuse": "java_misuse_sink",
        "mixed": "java_mixed_sink",
        "unknown": "java_unresolved_sink",
    }[label]


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "sink"


def _line_column(source: str, index: int) -> tuple[int | None, int | None]:
    if index < 0:
        return None, None
    line = source.count("\n", 0, index) + 1
    column = index - source.rfind("\n", 0, index)
    return line, column


def _algorithm_resolution_has_runtime_input(
    analyzer: SourceAnalyzer,
    expression: str,
    index: int,
    resolved: ResolvedValue,
) -> bool:
    """Recover runtime provenance hidden by a finite algorithm whitelist."""

    if resolved.origin == "runtime_input" or any(
        step.result == "runtime_input" for step in resolved.steps
    ):
        return True
    candidates = [expression]
    candidates.extend(
        step.expression
        for step in resolved.steps
        if step.expression != expression
    )
    for candidate in dict.fromkeys(candidates):
        if len(candidate) > 256:
            continue
        try:
            origin = analyzer.resolve_origin(candidate, index)
        except Exception:
            continue
        if origin.origin == "runtime_input" or any(
            step.result == "runtime_input" for step in origin.steps
        ):
            return True
    return False


def _password_label_with_reachability(
    analyzer: SourceAnalyzer,
    resolved: ResolvedValue,
) -> tuple[ProgramLabel, str, str]:
    """Ignore runtime alternatives that are proven unreachable at the sink."""

    label = _password_label(resolved)
    if (
        label[0] != "mixed"
        or _password_resolution_has_secure_random_alternative(resolved)
    ):
        return label
    runtime_steps = tuple(
        step
        for step in resolved.steps
        if step.result == "runtime_input"
    )
    if runtime_steps and all(
        not analyzer.position_is_reachable(step.span.start)
        for step in runtime_steps
    ):
        return (
            "misuse",
            "hardcoded_password",
            "the only reachable password source is constant or predictable",
        )
    return label


def _password_resolution_has_runtime_alternative(
    resolved: ResolvedValue,
) -> bool:
    if any(step.result == "runtime_input" for step in resolved.steps):
        return True
    if any(
        step.action == "resolve enclosing string equality branch"
        and step.result == "constant"
        for step in resolved.steps
    ):
        return False
    return any(
        re.search(
            r"\b(?:process\s*\.\s*(?:env|argv|stdin)|"
            r"Deno\s*\.\s*env|"
            r"Environment\s*\.\s*GetEnvironmentVariable|"
            r"System\s*\.\s*getenv|"
            r"os\s*\.\s*(?:environ|getenv)|"
            r"sys\s*\.\s*argv|"
            r"(?:input|prompt)\s*\(|"
            r"Console\s*\.\s*ReadLine\s*\()",
            step.expression,
        )
        is not None
        for step in resolved.steps
    )


def _preserve_called_python_function_findings(
    analyzer: SourceAnalyzer,
    lexical_findings: Iterable[DetectorFinding],
    active_findings: Iterable[DetectorFinding],
) -> list[DetectorFinding]:
    """Retain a definite locally selected sink in a syntactically called method.

    Free-form generations sometimes contain malformed outer orchestration while
    still expressing the requested native operation in a method that is called
    elsewhere in the source. The benchmark contract is static and accepts that
    operation sequence. Local branch reachability still prevents dormant
    alternatives in the same method from becoming evidence.
    """

    records = list(active_findings)
    if not isinstance(analyzer, PythonAnalyzer):
        return records
    signatures = {
        (item.span.start, item.span.end, item.expression, item.label)
        for item in records
    }
    records.extend(
        item
        for item in lexical_findings
        if item.label in {"secure", "misuse"}
        and analyzer.containing_function_is_called(item.span.start)
        and analyzer.position_is_locally_reachable(item.span.start)
        and (item.span.start, item.span.end, item.expression, item.label)
        not in signatures
    )
    return records


def _python_canonical_detector_call_name(
    analyzer: SourceAnalyzer,
    call: CallExpression,
) -> str:
    aliases = getattr(analyzer, "import_aliases", {})
    root, separator, suffix = call.callee.partition(".")
    return aliases.get(root, root) + (separator + suffix if separator else "")


def _salt_resolution_is_password_derived(resolved: ResolvedValue) -> bool:
    """Recognize salts deterministically copied or sliced from a password.

    Runtime input is ordinarily unknown for M06 because freshness cannot be
    established statically.  A salt made from the password itself is different:
    its value is coupled to the password and therefore cannot be independently
    fresh.  The resolver records helper returns and caller bindings in the
    resolution trace, which lets this rule cover direct and helper-mediated
    slices without treating arbitrary runtime salt input as misuse.
    """

    if resolved.origin != "runtime_input":
        return False
    password_reference = (
        r"(?:[A-Za-z_][A-Za-z0-9_]*\s*\.\s*)*"
        r"(?:password|passphrase|passwd|pwd)"
    )
    for step in resolved.steps:
        expression = step.expression.strip()
        if step.action.startswith("resolve assignment") and re.fullmatch(
            password_reference,
            expression,
            flags=re.IGNORECASE,
        ):
            return True
        if step.action in {
            "unwrap encoding conversion",
            "resolve indexed container element",
        } and re.match(
            rf"{password_reference}\b",
            expression,
            flags=re.IGNORECASE,
        ):
            return True
    return False


def _typescript_terminating_array_guard_values(
    source: str,
    target_name: str,
) -> tuple[str, ...]:
    pattern = re.compile(
        rf"\bif\s*\(\s*!\s*\[(?P<values>[^\]]+)\]"
        rf"\s*\.\s*includes\s*\(\s*{re.escape(target_name)}\s*\)\s*\)",
        flags=re.DOTALL,
    )
    for match in pattern.finditer(source):
        if not _typescript_guard_branch_terminates(
            source[match.end() : match.end() + 600]
        ):
            continue
        values: list[str] = []
        for item in match.group("values").split(","):
            literal = re.fullmatch(
                r"\s*([\"'])(?P<value>[^\"']+)\1\s*",
                item,
            )
            if literal is None:
                values = []
                break
            values.append(literal.group("value"))
        if values:
            return tuple(dict.fromkeys(values))
    return ()


def _typescript_terminating_comparison_guard_values(
    source: str,
    target_name: str,
) -> tuple[str, ...]:
    """Collect values retained by ``x !== a && x !== b`` guards."""

    target = re.escape(target_name)
    reference = rf"(?:[A-Za-z_$][A-Za-z0-9_$]*\s*\.\s*)*{target}"
    clause = re.compile(
        rf"\s*\(?\s*(?:"
        rf"{reference}\s*!==?\s*([\"'])(?P<right>[^\"']+)\1"
        rf"|([\"'])(?P<left>[^\"']+)\3\s*!==?\s*{reference}"
        rf")\s*\)?\s*"
    )
    for match in re.finditer(r"\bif\s*\(", source):
        opening = source.find("(", match.start(), match.end())
        closing = _matching_c_style_parenthesis(source, opening)
        if closing == -1:
            continue
        condition = source[opening + 1 : closing]
        parts = condition.split("&&")
        values: list[str] = []
        for part in parts:
            item = clause.fullmatch(part)
            if item is None:
                values = []
                break
            values.append(item.group("right") or item.group("left"))
        if not values:
            continue
        if not _typescript_guard_branch_terminates(
            source[closing + 1 : closing + 600]
        ):
            continue
        return tuple(dict.fromkeys(values))
    return ()
