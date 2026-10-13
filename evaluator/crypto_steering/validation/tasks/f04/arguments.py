"""F4 Rule 5: validate signature-call argument roles."""

from __future__ import annotations

from crypto_steering.validation.models import TaskValidationResult
from crypto_steering.validation.tasks.shared.results import (
    java_task_result,
    task_failure,
)
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.contracts import (
    _family_argument_contract_failure,
)
from crypto_steering.validation.tasks.f04.api_family import Analysis, required_evidence
from crypto_steering.validation.tasks.f04.operations import success_result
from crypto_steering.validation.tasks.shared.python_dataflow import (
    bytes_passed_to_encoded_parameter,
)


def validate_rule(
    context: TaskContext, analysis: Analysis
) -> TaskValidationResult | None:
    if context.language == "java":
        reason = _family_argument_contract_failure(
            context.language, context.source, family="F4"
        )
        if reason is None:
            return None
        return java_task_result(
            context,
            success=False,
            failure_code="target_argument_contract_invalid",
            reason=reason,
        )
    if any(item is None for item, _ in required_evidence(analysis)):
        return None
    if context.language == "python":
        conflict = bytes_passed_to_encoded_parameter(context.source)
        if conflict is not None:
            function, parameter = conflict
            return task_failure(
                success_result(context, analysis),
                "target_argument_contract_invalid",
                f"{function} receives bytes for {parameter}, then calls str.encode on it",
            )
    reason = _family_argument_contract_failure(
        context.language, context.source, family="F4"
    )
    if reason is None:
        return None
    return task_failure(
        success_result(context, analysis),
        "target_argument_contract_invalid",
        reason,
    )
