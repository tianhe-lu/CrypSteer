"""F1 Rule 6: require one connected symmetric-operation route."""

from __future__ import annotations

from crypto_steering.validation.java.task_analysis import (
    _receiver_method_calls,
    _receivers_call_method,
)
from crypto_steering.validation.models import TaskValidationResult
from crypto_steering.validation.tasks.shared.results import java_task_result, task_failure
from crypto_steering.validation.tasks.completion import (
    symmetric_direction_completion_failure,
)
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.f01.api_family import Analysis
from crypto_steering.validation.tasks.f01.arguments import initialized_receivers


def validate_rule(
    context: TaskContext, analysis: Analysis
) -> TaskValidationResult | None:
    if context.language == "java":
        receivers = initialized_receivers(analysis)
        if receivers and not _receivers_call_method(
            analysis.source, set(receivers), "doFinal"
        ):
            # Rule 4 already handles the absence of any terminal call.
            if _receiver_method_calls(analysis.source, "doFinal"):
                return java_task_result(
                    context,
                    success=False,
                    failure_code="target_operation_sequence_invalid",
                    reason="F1 requires Cipher.doFinal after valid Cipher initialization",
                )
            return None
        # Java records its route through receiver initialization and doFinal
        # evidence rather than the generic encrypt/decrypt fields used by the
        # other language analyzers. Reaching this point means that the Java
        # route is connected.
        return None

    reason = _family_route_failure(context, analysis)
    if reason is None:
        return None
    from crypto_steering.validation.tasks.f01.operations import success_result

    base = success_result(context, analysis)
    return task_failure(base, "target_operation_sequence_invalid", reason)


def _family_route_failure(
    context: TaskContext, analysis: Analysis
) -> str | None:
    """Require one connected direction without imposing a T-specific round trip."""

    failures = tuple(
        symmetric_direction_completion_failure(
            context.language,
            context.source,
            family="F1",
            direction=direction,
        )
        for direction, evidence in (
            ("encrypt", analysis.encrypt),
            ("decrypt", analysis.decrypt),
        )
        if evidence is not None
    )
    if any(reason is None for reason in failures):
        return None
    return failures[0] if failures else "no connected symmetric-cipher route"
