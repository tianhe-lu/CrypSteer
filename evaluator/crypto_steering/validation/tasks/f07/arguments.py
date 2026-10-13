"""F7 Rule 5: validate password-KDF argument roles."""

from __future__ import annotations

from crypto_steering.validation.models import TaskValidationResult
from crypto_steering.validation.tasks.shared.results import java_task_result, task_failure
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.contracts import (
    _family_argument_contract_failure,
)
from crypto_steering.validation.tasks.f07.api_family import Analysis


def validate_java_constructor(
    context: TaskContext, analysis: Analysis
) -> TaskValidationResult | None:
    if context.language != "java" or analysis.valid_pbe_calls:
        return None
    return java_task_result(
        context,
        success=False,
        failure_code="target_argument_contract_invalid",
        reason=(
            "F7 requires PBEKeySpec password, salt, and iteration arguments "
            "in usable constructor positions"
        ),
    )


def validate_java_terminal(
    context: TaskContext, analysis: Analysis
) -> TaskValidationResult | None:
    if (
        context.language != "java"
        or not analysis.generate_secret_calls
        or analysis.valid_generate_secret
    ):
        return None
    return java_task_result(
        context,
        success=False,
        failure_code="target_argument_contract_invalid",
        reason=(
            "F7 requires generateSecret with a PBEKeySpec/KeySpec derived "
            "from PBEKeySpec"
        ),
    )


def validate_rule(
    context: TaskContext,
    analysis: Analysis,
    result: TaskValidationResult,
) -> TaskValidationResult | None:
    if context.language == "java":
        return None
    if analysis.argument_contract is None:
        missing = ", ".join(analysis.missing_arguments) or "required arguments"
        return task_failure(
            result,
            "target_argument_contract_invalid",
            f"PBKDF2 is missing or misplaces {missing}",
        )
    # Compilers cover call shapes for C# and TypeScript.  The family check
    # below handles source-proven semantic argument failures that still compile,
    # while Python also needs parse-only native-call inspection.
    reason = _family_argument_contract_failure(
        context.language, context.source, family="F7"
    )
    if reason is None:
        return None
    return task_failure(result, "target_argument_contract_invalid", reason)
