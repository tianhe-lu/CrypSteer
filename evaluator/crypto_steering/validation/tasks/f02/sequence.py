"""F2 Rule 6: require a connected CBC route."""

from __future__ import annotations

import re

from crypto_steering.validation.java.source import iter_methods
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
from crypto_steering.validation.tasks.f02.api_family import Analysis
from crypto_steering.validation.tasks.f02.arguments import resolve_initialization
from crypto_steering.validation.tasks.f02.operations import success_result


def validate_rule(
    context: TaskContext, analysis: Analysis
) -> TaskValidationResult | None:
    if context.language == "java":
        receivers = resolve_initialization(analysis).receivers
        if receivers and not _receivers_call_method(
            analysis.source, set(receivers), "doFinal"
        ):
            if _receiver_method_calls(analysis.source, "doFinal"):
                return java_task_result(
                    context,
                    success=False,
                    failure_code="target_operation_sequence_invalid",
                    reason="F2 requires Cipher.doFinal after valid CBC initialization",
                )
            return None
        # Java uses receiver-based CBC evidence. The generic direction fields
        # below are intentionally empty for Java, so a successful receiver
        # route must return here.
        return None
    reason = _family_route_failure(context, analysis)
    if reason is None:
        return None
    return task_failure(
        success_result(context, analysis),
        "target_operation_sequence_invalid",
        reason,
    )


def _family_route_failure(
    context: TaskContext, analysis: Analysis
) -> str | None:
    failures = tuple(
        symmetric_direction_completion_failure(
            context.language,
            context.source,
            family="F2",
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
    return failures[0] if failures else "no connected CBC route"


def encrypt_path_is_explicitly_non_cbc(source: str) -> bool:
    """Find an encryption helper whose literal transformation is not CBC."""

    methods = tuple(
        method for method in iter_methods(source)
        if "Cipher.ENCRYPT_MODE" in method.text
    )
    if not methods:
        return False
    saw_explicit = False
    for method in methods:
        transformations = re.findall(
            r"\bCipher\s*\.\s*getInstance\s*\(\s*[\"']([^\"']+)[\"']",
            method.text,
        )
        if not transformations:
            return False
        saw_explicit = True
        if any("CBC" in value.upper() for value in transformations):
            return False
    return saw_explicit
