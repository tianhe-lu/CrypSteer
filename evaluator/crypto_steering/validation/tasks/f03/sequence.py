"""F3 Rule 6: require a connected RSA operation route."""

from __future__ import annotations

from crypto_steering.validation.java.task_analysis import (
    _receiver_method_calls,
    _receivers_call_method,
)
from crypto_steering.validation.models import TaskValidationResult
from crypto_steering.validation.tasks.shared.results import java_task_result
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.f03.api_family import Analysis
from crypto_steering.validation.tasks.f03.arguments import initialized_receivers


def validate_rule(
    context: TaskContext, analysis: Analysis
) -> TaskValidationResult | None:
    if context.language != "java":
        return None
    receivers = initialized_receivers(analysis)
    if receivers and not _receivers_call_method(
        analysis.source, set(receivers), "doFinal"
    ):
        if _receiver_method_calls(analysis.source, "doFinal"):
            return java_task_result(
                context,
                success=False,
                failure_code="target_operation_sequence_invalid",
                reason="F3 requires Cipher.doFinal after valid RSA initialization",
            )
        return None
    return None
