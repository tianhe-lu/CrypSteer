"""F6 Rule 6: require one connected MAC generation or verification route."""

from __future__ import annotations

from crypto_steering.validation.java.task_analysis import _receiver_method_calls, _receivers_call_method
from crypto_steering.validation.models import TaskValidationResult
from crypto_steering.validation.tasks.shared.results import java_task_result
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.f06.api_family import Analysis
from crypto_steering.validation.tasks.f06.arguments import initialized_receivers
from crypto_steering.validation.tasks.macs import _typescript_reuses_finalized_hmac
from crypto_steering.validation.tasks.shared.results import task_failure


def validate_java_connection(
    context: TaskContext,
    analysis: Analysis,
    result: TaskValidationResult,
) -> TaskValidationResult | None:
    if context.language != "java":
        return None
    receivers = initialized_receivers(analysis)
    if receivers and not _receivers_call_method(
        analysis.source, set(receivers), "doFinal"
    ) and _receiver_method_calls(analysis.source, "doFinal"):
        return java_task_result(
            context,
            success=False,
            failure_code="target_operation_sequence_invalid",
            reason="F6 requires Mac.doFinal after valid Mac initialization",
        )
    return None


def validate_rule(
    context: TaskContext,
    analysis: Analysis,
    result: TaskValidationResult,
) -> TaskValidationResult | None:
    """Generation or native verification completes F6 for every T variant."""

    if (
        context.language == "typescript"
        and _typescript_reuses_finalized_hmac(context.source)
    ):
        return task_failure(
            result,
            "target_operation_sequence_invalid",
            "the HMAC is finalized before its message update and then reused",
        )
    return None
