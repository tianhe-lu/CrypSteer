"""F8 Rule 5: validate password and serialization argument roles."""

from __future__ import annotations

from crypto_steering.validation.java.task_analysis import _first_bad_keystore_password_call
from crypto_steering.validation.models import TaskValidationResult
from crypto_steering.validation.tasks.shared.results import java_task_result, task_failure
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.contracts import _family_argument_contract_failure
from crypto_steering.validation.tasks.f08.api_family import Analysis


def validate_java_rule(
    context: TaskContext, analysis: Analysis
) -> TaskValidationResult | None:
    if context.language != "java":
        return None
    call = _first_bad_keystore_password_call(
        analysis.source, list(analysis.assignments)
    )
    if call is None:
        return None
    return java_task_result(
        context,
        success=False,
        failure_code="target_argument_contract_invalid",
        reason=f"F8 password argument for {call.name} must be char[]-typed",
    )


def validate_rule(
    context: TaskContext,
    analysis: Analysis,
    result: TaskValidationResult,
) -> TaskValidationResult | None:
    if context.language == "java":
        return None
    reason = _family_argument_contract_failure(
        context.language, context.source, family="F8"
    )
    if reason is None:
        return None
    return task_failure(result, "target_argument_contract_invalid", reason)
