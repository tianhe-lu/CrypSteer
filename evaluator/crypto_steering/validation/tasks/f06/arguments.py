"""F6 Rule 5: validate MAC key and data argument roles."""

from __future__ import annotations

from crypto_steering.validation.java.task_analysis import _is_key_expression
from crypto_steering.validation.models import TaskValidationResult
from crypto_steering.validation.tasks.shared.results import java_task_result, task_failure
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.contracts import _family_argument_contract_failure
from crypto_steering.validation.tasks.f06.api_family import Analysis
from crypto_steering.validation.tasks.shared.python_dataflow import bytes_passed_to_encoded_parameter


def initialized_receivers(analysis: Analysis) -> frozenset[str]:
    return frozenset(
        receiver
        for receiver, call in analysis.candidate_initializations
        if call.arguments
        and _is_key_expression(analysis.source, call.arguments[0].text, call.start)
    )


def validate_java_rule(
    context: TaskContext, analysis: Analysis
) -> TaskValidationResult | None:
    if context.language != "java" or not analysis.candidate_initializations:
        return None
    if initialized_receivers(analysis):
        return None
    return java_task_result(
        context,
        success=False,
        failure_code="target_argument_contract_invalid",
        reason="F6 requires Mac.init with key material",
    )


def validate_python_encoding(
    context: TaskContext,
    analysis: Analysis,
    result: TaskValidationResult,
) -> TaskValidationResult | None:
    if context.language != "python":
        return None
    conflict = bytes_passed_to_encoded_parameter(context.source)
    if conflict is None:
        return None
    function, parameter = conflict
    return task_failure(
        result,
        "target_argument_contract_invalid",
        f"{function} receives bytes for {parameter}, then calls str.encode on it",
    )


def validate_rule(
    context: TaskContext,
    analysis: Analysis,
    result: TaskValidationResult,
) -> TaskValidationResult | None:
    reason = _family_argument_contract_failure(
        context.language, context.source, family="F6"
    )
    if reason is None:
        return None
    return task_failure(result, "target_argument_contract_invalid", reason)
