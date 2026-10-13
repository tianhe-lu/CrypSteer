"""F7 Rule 3: recognize supported password-based key derivation APIs."""

from __future__ import annotations

from dataclasses import dataclass

from crypto_steering.validation.java.source import find_calls, strip_java_comments
from crypto_steering.validation.java.task_analysis import (
    _has_secret_key_factory_pbkdf2,
    _is_pbe_key_spec_expression,
    _is_valid_f7_pbe_key_spec_call,
    _pbe_spec_variables,
)
from crypto_steering.validation.models import TaskEvidence, TaskValidationResult
from crypto_steering.validation.tasks.shared.results import java_task_result, require_analyzer
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.evidence import (
    _first_evidence,
    _pbkdf2_setup_evidence,
)


@dataclass(frozen=True)
class Analysis:
    source: str
    setup: TaskEvidence | None = None
    argument_contract: TaskEvidence | None = None
    terminal: TaskEvidence | None = None
    missing_arguments: tuple[str, ...] = ()
    pbe_calls: tuple[object, ...] = ()
    valid_pbe_calls: tuple[object, ...] = ()
    supported_factory: bool = False
    generate_secret_calls: tuple[object, ...] = ()
    valid_generate_secret: tuple[object, ...] = ()


def analyze(
    context: TaskContext,
    *,
    allow_general_password_kdf: bool,
) -> Analysis:
    if context.language == "java":
        source = strip_java_comments(context.source)
        pbe_calls = tuple(
            find_calls(
                source,
                r"\bnew\s+(?:javax\.crypto\.spec\.)?PBEKeySpec\s*\(",
                "PBEKeySpec",
            )
        )
        valid_pbe = tuple(
            call for call in pbe_calls
            if _is_valid_f7_pbe_key_spec_call(source, call)
        )
        generate = tuple(
            find_calls(
                source,
                r"\.\s*generateSecret\s*\(",
                "SecretKeyFactory.generateSecret",
            )
        )
        variables = _pbe_spec_variables(source)
        valid_generate = tuple(
            call for call in generate
            if call.arguments
            and _is_pbe_key_spec_expression(
                source, call.arguments[0].text, call.start, variables
            )
        )
        return Analysis(
            source=source,
            pbe_calls=pbe_calls,
            valid_pbe_calls=valid_pbe,
            supported_factory=_has_secret_key_factory_pbkdf2(
                source, allow_general_pbe=allow_general_password_kdf
            ),
            generate_secret_calls=generate,
            valid_generate_secret=valid_generate,
        )

    analyzer = require_analyzer(context)
    setup, argument_contract, missing = _pbkdf2_setup_evidence(analyzer)
    terminal = _first_evidence(
        analyzer,
        (
            "pbkdf2", "pbkdf2Sync", "pbkdf2_hmac", "Pbkdf2", "GetBytes",
            "derive", "verify", "generateSecret",
        ),
    )
    if terminal is None and setup is not None and setup.operation not in {
        "PBKDF2HMAC", "Rfc2898DeriveBytes", "PBEKeySpec"
    }:
        terminal = setup
    if (
        terminal is None
        and context.language == "python"
        and setup is not None
        and argument_contract is not None
    ):
        for call in analyzer.calls("derive", "verify"):
            if analyzer.containing_function_name(call.start) is not None:
                terminal = TaskEvidence(
                    operation=call.simple_name,
                    expression=analyzer.source[call.start : call.end],
                    span=analyzer.source_span(call.start, call.end),
                )
                break
    return Analysis(
        source=context.source,
        setup=setup,
        argument_contract=argument_contract,
        terminal=terminal,
        missing_arguments=missing,
    )


def validate_presence(
    context: TaskContext, analysis: Analysis
) -> TaskValidationResult | None:
    if context.language == "java":
        if analysis.pbe_calls:
            return None
        return java_task_result(
            context,
            success=False,
            failure_code="target_api_not_found",
            reason="F7 requires actual PBEKeySpec constructor usage",
        )
    if analysis.setup is not None:
        return None
    return TaskValidationResult(
        language=context.language,
        task_id=context.task_id,
        success=False,
        reason_codes=("target_api_not_found",),
        reason="missing supported PBKDF2 construction",
    )


def validate_rule(
    context: TaskContext, analysis: Analysis
) -> TaskValidationResult | None:
    if context.language != "java" or analysis.supported_factory:
        return None
    return java_task_result(
        context,
        success=False,
        failure_code="target_wrong_api_family",
        reason="F7 requires a supported PBKDF2 SecretKeyFactory algorithm",
    )
