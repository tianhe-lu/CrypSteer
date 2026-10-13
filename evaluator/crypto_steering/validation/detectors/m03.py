"""M03 CBC-IV detector."""

from __future__ import annotations

from dataclasses import replace
import re

from crypto_steering.validation.java.calls import (
    constructor_calls,
    receiver_method_calls,
    source_contains_any_token,
    statement_text,
    static_method_calls,
)
from crypto_steering.validation.java.detection import (
    JavaEvidence,
    evidence_result,
    parse_failure_result,
)
from crypto_steering.validation.java.lightweight_ast import (
    find_method_calls,
    method_containing_index,
)
from crypto_steering.validation.java.security_value_flow import (
    SecurityValueFlow,
    SourceClassification,
)
from crypto_steering.validation.java.static_analysis import simple_variable
from crypto_steering.validation.models import (
    DetectionResult,
    DetectorFinding,
    Language,
    combine_detector_findings,
)
from crypto_steering.validation.source.index import (
    CStyleAnalyzer,
    JavaAnalyzer,
    SourceAnalyzer,
)
from crypto_steering.validation.detectors.apis import (
    _csharp_provider_property_terminal_direction,
    _csharp_uses_cbc,
)
from crypto_steering.validation.detectors.core import (
    _origin_finding,
    _finding_from_resolved,
    _fresh_value_label,
    _csharp_iv_origin,
    _with_typescript_cipher_mode_fragments,
    _result,
    _suppress_unresolved_uncalled_helpers,
    _with_operation_direction,
    _suppress_unresolved_decryption_reuse,
    _suppress_proven_inactive_findings,
    _preserve_definite_inactive_findings,
    _direct_call_finding,
    _normalize_algorithm,
)
from crypto_steering.validation.detectors.engine import execute_detector


def _detect_m03(analyzer: SourceAnalyzer, task_id: str | None, modern: int) -> DetectionResult:
    if isinstance(analyzer, JavaAnalyzer):
        return _detect_java_m03(analyzer, task_id)
    del task_id, modern
    findings: list[DetectorFinding] = []
    if analyzer.language == "typescript":
        for call in analyzer.calls("createCipheriv", "createDecipheriv"):
            if len(call.arguments) < 3:
                continue
            algorithm = _with_typescript_cipher_mode_fragments(
                analyzer.resolve_strings(call.arguments[0], call.start)
            )
            if not any("CBC" in _normalize_algorithm(value) for value in algorithm.values):
                continue
            finding = _origin_finding(
                analyzer,
                "M03",
                call,
                call.arguments[2],
                "cbc_iv",
                _fresh_value_label,
            )
            findings.append(_with_operation_direction(analyzer, finding, call.start, call.simple_name))
    elif analyzer.language == "csharp":
        if not _csharp_uses_cbc(analyzer):
            return _result(analyzer, "M03", findings, "no CBC encryption flow found")
        if isinstance(analyzer, CStyleAnalyzer):
            for assignment in analyzer.assignments:
                compact_assignment_name = re.sub(r"\s+", "", assignment.name)
                terminal_direction = _csharp_provider_property_terminal_direction(
                    analyzer,
                    compact_assignment_name,
                    assignment.end,
                )
                if (
                    assignment.name.rsplit(".", 1)[-1] == "IV"
                    and "." in assignment.name
                    and terminal_direction is not None
                ):
                    resolved = _csharp_iv_origin(
                        analyzer,
                        assignment.expression,
                        assignment.start,
                    )
                    finding = _finding_from_resolved(
                        analyzer, "M03", "cbc_iv_property", "SymmetricAlgorithm.IV",
                        assignment.expression, assignment.start, assignment.end, resolved, _fresh_value_label,
                    )
                    findings.append(
                        replace(
                            finding,
                            metadata={
                                **finding.metadata,
                                "operation_direction": terminal_direction,
                            },
                        )
                    )
        for call in analyzer.calls("CreateEncryptor", "CreateDecryptor"):
            if len(call.arguments) >= 2:
                expression = call.arguments[1]
                resolved = _csharp_iv_origin(analyzer, expression, call.start)
                finding = _finding_from_resolved(
                    analyzer,
                    "M03",
                    "cbc_iv",
                    call.callee,
                    expression,
                    call.start,
                    call.end,
                    resolved,
                    _fresh_value_label,
                )
                findings.append(_with_operation_direction(analyzer, finding, call.start, call.simple_name))
        has_encryption_iv = any(
            finding.metadata.get("operation_direction") != "decrypt"
            for finding in findings
        )
        if not has_encryption_iv:
            for call in analyzer.calls("CreateEncryptor", "CreateDecryptor"):
                if call.arguments:
                    continue
                finding = _direct_call_finding(
                    analyzer,
                    "M03",
                    call,
                    "secure",
                    "provider_generated_iv",
                    "parameterless AES transform uses the provider-generated IV",
                )
                findings.append(
                    _with_operation_direction(
                        analyzer,
                        finding,
                        call.start,
                        call.simple_name,
                    )
                )
    elif analyzer.language == "python":
        for call in analyzer.calls("CBC"):
            if call.arguments:
                finding = _origin_finding(
                    analyzer,
                    "M03",
                    call,
                    call.arguments[0],
                    "cbc_iv",
                    _fresh_value_label,
                )
                findings.append(_with_operation_direction(analyzer, finding, call.start, call.simple_name))
    findings = _suppress_unresolved_decryption_reuse(findings)
    lexical_findings = tuple(findings)
    findings = _suppress_proven_inactive_findings(analyzer, findings)
    findings = _suppress_unresolved_uncalled_helpers(analyzer, findings)
    findings = _preserve_definite_inactive_findings(
        analyzer,
        lexical_findings,
        findings,
    )
    return _result(analyzer, "M03", findings, "no CBC IV sink found")


def _detect_java_m03(
    analyzer: JavaAnalyzer,
    task_id: str | None,
) -> DetectionResult:
    del task_id
    if _java_has_only_decryption_direction(analyzer.source):
        return combine_detector_findings(
            "M03",
            "java",
            (),
            no_sink_reason="no CBC encryption IV choice found",
        )
    if not _java_m03_source_may_contain_sink(analyzer.source):
        return evidence_result(
            analyzer,
            "M03",
            (),
            no_sink_reason="no CBC IV-sensitive sink found",
        )
    if not analyzer.flow.ast_context.parse_success:
        return parse_failure_result(analyzer, "M03")
    evidence = _java_m03_evidence(analyzer)
    primary = [
        item
        for item in evidence
        if not item.reason.startswith("CBC decrypt IV")
        and item.sink_name != "Cipher.init provider-generated IV"
    ]
    return evidence_result(
        analyzer,
        "M03",
        primary or evidence,
        no_sink_reason="no CBC IV-sensitive sink found",
    )


def _java_m03_source_may_contain_sink(source: str) -> bool:
    return source_contains_any_token(
        source,
        {"IvParameterSpec", "Cipher", "CBC"},
    )


def _java_m03_evidence(analyzer: JavaAnalyzer) -> list[JavaEvidence]:
    source = analyzer.source
    flow = analyzer.flow
    evidence: list[JavaEvidence] = []
    if not _java_contains_cipher_mode(source, "CBC"):
        return evidence
    for call in constructor_calls(source, "IvParameterSpec"):
        if not call.arguments:
            continue
        argument = call.arguments[0]
        context = _java_iv_usage_context(
            source,
            call.start,
            argument.text,
            flow,
        )
        if context == "unknown":
            continue
        spec_variable = _java_iv_assigned_variable(
            statement_text(source, call.start)
        )
        if (
            context == "encrypt"
            and spec_variable is not None
            and not _java_iv_spec_reaches_active_encryption(
                source,
                flow,
                spec_variable,
                call.start,
            )
        ):
            continue
        if context == "decrypt_only":
            classification = flow.classify(
                argument.text,
                argument.start,
                "iv",
            )
            if classification.label == "misuse":
                classification = SourceClassification(
                    "misuse",
                    "CBC decrypt IV source resolves to a predictable value: "
                    f"{classification.reason}",
                )
            else:
                classification = SourceClassification(
                    "secure",
                    "CBC decrypt IV is consumed from ciphertext context and "
                    "is not generated here",
                )
        elif context == "encrypt" and _java_iv_reuses_plaintext(
            source,
            flow,
            argument.text,
            call.start,
        ):
            classification = SourceClassification(
                "misuse",
                "CBC encryption IV is deterministically derived from plaintext",
            )
        else:
            classification = flow.classify(
                argument.text,
                argument.start,
                "iv",
            )
        evidence.append(
            JavaEvidence(
                sink_name="new IvParameterSpec CBC IV",
                expression=argument.text,
                label=classification.label,
                reason=classification.reason,
                start=argument.start,
                end=argument.end,
            )
        )

    cipher_receivers = _java_cipher_receivers(source)
    for receiver, call in receiver_method_calls(source, "init"):
        if receiver not in cipher_receivers or len(call.arguments) != 2:
            continue
        if "Cipher.ENCRYPT_MODE" not in call.arguments[0].text:
            continue
        evidence.append(
            JavaEvidence(
                sink_name="Cipher.init provider-generated IV",
                expression=source[call.start : call.end],
                label="secure",
                reason=(
                    "CBC encryption init omits explicit IV, so the provider "
                    "generates the IV"
                ),
                start=call.start,
                end=call.end,
            )
        )
    return evidence


def _java_contains_cipher_mode(source: str, mode_name: str) -> bool:
    mode = mode_name.upper()
    if re.search(
        rf"\bCipher\s*\.\s*getInstance\s*\(\s*\"[^\"]*"
        rf"{re.escape(mode_name)}[^\"]*\"\s*\)",
        source,
        flags=re.IGNORECASE,
    ):
        return True
    constants = {
        match.group("name"): match.group("value")
        for match in re.finditer(
            r"\b(?:public|private|protected|static|final|\s)*\s*"
            r"(?:String|java\.lang\.String)\s+"
            r"(?P<name>[A-Za-z_]\w*)\s*=\s*"
            r'"(?P<value>(?:\\.|[^"\\])*)"\s*;',
            source,
            flags=re.DOTALL,
        )
    }
    for call in static_method_calls(source, "Cipher", "getInstance"):
        if call.arguments and mode in _java_constant_string_expression(
            call.arguments[0].text,
            constants,
        ).upper():
            return True
    return False


def _java_constant_string_expression(
    expression: str,
    constants: dict[str, str],
) -> str:
    pieces: list[str] = []
    for item in _java_split_string_concat(expression):
        item = item.strip()
        literal = re.fullmatch(
            r'"((?:\\.|[^"\\])*)"',
            item,
            flags=re.DOTALL,
        )
        if literal:
            pieces.append(literal.group(1))
            continue
        variable = re.fullmatch(r"[A-Za-z_]\w*", item)
        if variable and variable.group(0) in constants:
            pieces.append(constants[variable.group(0)])
            continue
        pieces.append(item)
    return "".join(pieces)


def _java_split_string_concat(expression: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(expression):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "([{":
            depth += 1
        elif char in ")]}" and depth:
            depth -= 1
        elif char == "+" and depth == 0:
            parts.append(expression[start:index])
            start = index + 1
    parts.append(expression[start:])
    return parts


def _java_iv_usage_context(
    source: str,
    call_start: int,
    expression: str,
    flow: SecurityValueFlow,
) -> str:
    statement = statement_text(source, call_start)
    if "Cipher.DECRYPT_MODE" in statement and "Cipher.ENCRYPT_MODE" not in statement:
        return "decrypt_only"
    if "Cipher.ENCRYPT_MODE" in statement:
        return "encrypt"

    method = method_containing_index(flow.methods, call_start)
    scope_end = method.body_end if method is not None else len(source)
    spec_variable = _java_iv_assigned_variable(statement)
    if spec_variable and _java_cipher_init_uses_variable(
        source,
        spec_variable,
        "ENCRYPT_MODE",
        start=call_start,
        end=scope_end,
    ):
        return "encrypt"
    if spec_variable and _java_method_argument_reaches_cipher_init(
        source,
        flow,
        spec_variable,
        "ENCRYPT_MODE",
        call_start,
    ):
        return "encrypt"
    if spec_variable and _java_cipher_init_uses_variable(
        source,
        spec_variable,
        "DECRYPT_MODE",
        start=call_start,
        end=scope_end,
    ):
        return "decrypt_only"
    if spec_variable and _java_method_argument_reaches_cipher_init(
        source,
        flow,
        spec_variable,
        "DECRYPT_MODE",
        call_start,
    ):
        return "decrypt_only"

    if method is not None:
        caller_context = _java_enclosing_method_mode_context(
            source,
            flow,
            method,
            call_start,
        )
        if caller_context != "unknown":
            return caller_context
        if _java_method_result_used_in_cipher_init(
            source,
            method.name,
            "ENCRYPT_MODE",
        ):
            return "encrypt"
        if _java_method_result_reaches_cipher_init(
            source,
            flow,
            method.name,
            "ENCRYPT_MODE",
            method,
        ):
            return "encrypt"
        if _java_method_result_used_in_cipher_init(
            source,
            method.name,
            "DECRYPT_MODE",
        ):
            return "decrypt_only"
        if _java_method_result_reaches_cipher_init(
            source,
            flow,
            method.name,
            "DECRYPT_MODE",
            method,
        ):
            return "decrypt_only"

    if _java_constructor_argument_reaches_cipher_init(
        source,
        flow,
        call_start,
        "ENCRYPT_MODE",
    ):
        return "encrypt"
    if _java_constructor_argument_reaches_cipher_init(
        source,
        flow,
        call_start,
        "DECRYPT_MODE",
    ):
        return "decrypt_only"
    variable = simple_variable(expression)
    if variable and _java_cipher_init_uses_variable(
        source,
        variable,
        "ENCRYPT_MODE",
        start=call_start,
        end=scope_end,
    ):
        return "encrypt"
    if variable and _java_cipher_init_uses_variable(
        source,
        variable,
        "DECRYPT_MODE",
        start=call_start,
        end=scope_end,
    ):
        return "decrypt_only"
    return "unknown"


def _java_enclosing_method_mode_context(
    source: str,
    flow: SecurityValueFlow,
    method,
    constructor_start: int,
) -> str:
    parameter_names = [parameter.name for parameter in method.params]
    if not parameter_names:
        return "unknown"
    mode_parameter_index: int | None = None
    for _receiver, init_call in receiver_method_calls(source, "init"):
        if not method.body_start <= init_call.start <= method.body_end:
            continue
        if not init_call.start <= constructor_start <= init_call.end:
            continue
        if not init_call.arguments:
            continue
        mode_variable = simple_variable(init_call.arguments[0].text)
        if mode_variable in parameter_names:
            mode_parameter_index = parameter_names.index(mode_variable)
            break
    if mode_parameter_index is None:
        return "unknown"
    directions: set[str] = set()
    for call in find_method_calls(source, method.name):
        if method.start <= call.start <= method.body_end:
            continue
        if len(call.arguments) != len(parameter_names):
            continue
        mode_argument = call.arguments[mode_parameter_index].text
        if re.search(r"\bCipher\s*\.\s*ENCRYPT_MODE\b", mode_argument):
            directions.add("encrypt")
        if re.search(r"\bCipher\s*\.\s*DECRYPT_MODE\b", mode_argument):
            directions.add("decrypt")
    if "encrypt" in directions:
        return "encrypt"
    if directions == {"decrypt"}:
        return "decrypt_only"
    return "unknown"


def _java_iv_assigned_variable(statement: str) -> str | None:
    match = re.search(
        r"(?:\bIvParameterSpec\s+)?([A-Za-z_]\w*)\s*=\s*[^;]*"
        r"new\s+(?:[\w.]+\.)?IvParameterSpec\s*\(",
        statement,
        flags=re.DOTALL,
    )
    return match.group(1) if match else None


def _java_iv_spec_reaches_active_encryption(
    source: str,
    flow: SecurityValueFlow,
    variable: str,
    constructor_start: int,
) -> bool:
    if _java_variable_reaches_active_encryption(
        source,
        variable,
        start=constructor_start,
        end=len(source),
    ):
        return True
    for method in flow.methods:
        parameter_names = [parameter.name for parameter in method.params]
        if not parameter_names:
            continue
        for call in find_method_calls(source, method.name):
            if method.body_start <= call.start <= method.body_end:
                continue
            if call.start <= constructor_start or len(call.arguments) != len(parameter_names):
                continue
            for argument_index, argument in enumerate(call.arguments):
                if simple_variable(argument.text) != variable:
                    continue
                if _java_variable_reaches_active_encryption(
                    source,
                    parameter_names[argument_index],
                    start=method.body_start,
                    end=method.body_end,
                ):
                    return True
    return False


def _java_variable_reaches_active_encryption(
    source: str,
    variable: str,
    *,
    start: int,
    end: int,
) -> bool:
    init_calls = tuple(receiver_method_calls(source, "init"))
    terminal_calls = (
        *receiver_method_calls(source, "update"),
        *receiver_method_calls(source, "doFinal"),
    )
    for receiver, call in init_calls:
        if not start <= call.start <= end or len(call.arguments) < 3:
            continue
        if "Cipher.ENCRYPT_MODE" not in call.arguments[0].text:
            continue
        if re.search(rf"\b{re.escape(variable)}\b", call.arguments[2].text) is None:
            continue
        next_init = min(
            (
                candidate.start
                for candidate_receiver, candidate in init_calls
                if candidate_receiver == receiver and call.start < candidate.start <= end
            ),
            default=end,
        )
        mutually_exclusive_reinit = bool(
            next_init < end
            and re.search(
                r"}\s*else\s+(?:if\b[\s\S]*?)?{\s*$",
                source[call.start:next_init],
            )
        )
        if any(
            terminal_receiver == receiver
            and call.start < terminal.start <= end
            and (terminal.start < next_init or mutually_exclusive_reinit)
            for terminal_receiver, terminal in terminal_calls
        ):
            return True
    return False


def _java_iv_reuses_plaintext(
    source: str,
    flow: SecurityValueFlow,
    expression: str,
    constructor_start: int,
) -> bool:
    method = method_containing_index(flow.methods, constructor_start)
    scope_start = method.body_start if method is not None else 0
    scope_end = method.body_end if method is not None else len(source)
    declared_names = {assignment.name for assignment in flow.assignments}
    declared_names.update(
        parameter.name
        for candidate_method in flow.methods
        for parameter in candidate_method.params
    )
    iv_names = set(re.findall(r"\b[A-Za-z_]\w*\b", expression)) & declared_names
    if not iv_names:
        return False
    for _receiver, terminal in receiver_method_calls(source, "doFinal"):
        if not scope_start <= terminal.start <= scope_end or not terminal.arguments:
            continue
        terminal_names = (
            set(re.findall(r"\b[A-Za-z_]\w*\b", terminal.arguments[0].text))
            & declared_names
        )
        if iv_names & terminal_names:
            return True
    return False


def _java_cipher_init_uses_variable(
    source: str,
    variable: str,
    mode: str,
    *,
    start: int = 0,
    end: int | None = None,
) -> bool:
    scope_end = end if end is not None else len(source)
    for _receiver, call in receiver_method_calls(source, "init"):
        if not start <= call.start <= scope_end or len(call.arguments) < 3:
            continue
        if re.search(
            rf"\bCipher\s*\.\s*{re.escape(mode)}\b",
            call.arguments[0].text,
        ) is None:
            continue
        if re.search(rf"\b{re.escape(variable)}\b", call.arguments[2].text) is None:
            continue
        if re.search(
            rf"(?<![.\w$]){re.escape(variable)}\s*=(?!=)",
            source[start:call.start],
        ) is None:
            return True
    return False


def _java_method_argument_reaches_cipher_init(
    source: str,
    flow: SecurityValueFlow,
    variable: str,
    mode: str,
    call_start: int,
) -> bool:
    for method in flow.methods:
        parameter_names = [parameter.name for parameter in method.params]
        if not parameter_names:
            continue
        for call in find_method_calls(source, method.name):
            if method.body_start <= call.start <= method.body_end:
                continue
            if call.start <= call_start or len(call.arguments) != len(parameter_names):
                continue
            for argument_index, argument in enumerate(call.arguments):
                if simple_variable(argument.text) != variable:
                    continue
                if _java_cipher_init_uses_variable(
                    source,
                    parameter_names[argument_index],
                    mode,
                    start=method.body_start,
                    end=method.body_end,
                ):
                    return True
    return False


def _java_constructor_argument_reaches_cipher_init(
    source: str,
    flow: SecurityValueFlow,
    constructor_start: int,
    mode: str,
) -> bool:
    for method in flow.methods:
        parameter_names = [parameter.name for parameter in method.params]
        if not parameter_names:
            continue
        for call in find_method_calls(source, method.name):
            if method.body_start <= call.start <= method.body_end:
                continue
            if not call.start <= constructor_start <= call.end:
                continue
            if len(call.arguments) != len(parameter_names):
                continue
            for argument_index, argument in enumerate(call.arguments):
                if not argument.start <= constructor_start <= argument.end:
                    continue
                if _java_cipher_init_uses_variable(
                    source,
                    parameter_names[argument_index],
                    mode,
                    start=method.body_start,
                    end=method.body_end,
                ):
                    return True
    return False


def _java_method_result_reaches_cipher_init(
    source: str,
    flow: SecurityValueFlow,
    method_name: str,
    mode: str,
    result_method,
) -> bool:
    for call in find_method_calls(source, method_name):
        if result_method.body_start <= call.start <= result_method.body_end:
            continue
        assigned_variable = _java_assigned_variable_from_call(
            statement_text(source, call.start),
            method_name,
        )
        if assigned_variable:
            call_method = method_containing_index(flow.methods, call.start)
            scope_end = call_method.body_end if call_method is not None else len(source)
            if _java_cipher_init_uses_variable(
                source,
                assigned_variable,
                mode,
                start=call.end,
                end=scope_end,
            ):
                return True
            if _java_method_argument_reaches_cipher_init(
                source,
                flow,
                assigned_variable,
                mode,
                call.start,
            ):
                return True
        if _java_call_result_argument_reaches_cipher_init(
            source,
            flow,
            call.start,
            mode,
        ):
            return True
    return False


def _java_assigned_variable_from_call(
    statement: str,
    method_name: str,
) -> str | None:
    match = re.search(
        rf"(?:\bfinal\s+)?(?:\b[A-Za-z_]\w*(?:<[^;=]+>)?"
        rf"(?:\[\])?\s+)?([A-Za-z_]\w*)\s*=\s*"
        rf"(?:(?:this|[A-Za-z_]\w*)\s*\.\s*)?"
        rf"{re.escape(method_name)}\s*\(",
        statement,
        flags=re.DOTALL,
    )
    return match.group(1) if match else None


def _java_call_result_argument_reaches_cipher_init(
    source: str,
    flow: SecurityValueFlow,
    call_start: int,
    mode: str,
) -> bool:
    for method in flow.methods:
        parameter_names = [parameter.name for parameter in method.params]
        if not parameter_names:
            continue
        for call in find_method_calls(source, method.name):
            if method.body_start <= call.start <= method.body_end:
                continue
            if not call.start <= call_start <= call.end:
                continue
            if len(call.arguments) != len(parameter_names):
                continue
            for argument_index, argument in enumerate(call.arguments):
                if not argument.start <= call_start <= argument.end:
                    continue
                if _java_cipher_init_uses_variable(
                    source,
                    parameter_names[argument_index],
                    mode,
                    start=method.body_start,
                    end=method.body_end,
                ):
                    return True
    return False


def _java_cipher_receivers(source: str) -> set[str]:
    return {
        match.group("name")
        for match in re.finditer(
            r"(?:\bCipher\s+)?(?P<name>[A-Za-z_]\w*)\s*=\s*"
            r"(?:[\w.]+\.)?Cipher\s*\.\s*getInstance\s*\(",
            source,
        )
    }


def _java_method_result_used_in_cipher_init(
    source: str,
    method_name: str,
    mode: str,
) -> bool:
    return bool(
        re.search(
            rf"\.init\s*\(\s*Cipher\.{mode}\s*,[^;]*\b"
            rf"{re.escape(method_name)}\s*\(",
            source,
            flags=re.DOTALL,
        )
    )


def _java_has_only_decryption_direction(source: str) -> bool:
    has_decrypt = bool(re.search(r"\bCipher\s*\.\s*DECRYPT_MODE\b", source))
    has_encrypt = bool(re.search(r"\bCipher\s*\.\s*ENCRYPT_MODE\b", source))
    return has_decrypt and not has_encrypt



MISUSE_ID = "M03"


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
        implementation=_detect_m03,
    )


__all__ = ("MISUSE_ID", "detect")
