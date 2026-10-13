"""F5 Rule 5: validate digest input argument structure."""

from __future__ import annotations

from crypto_steering.validation.models import TaskValidationResult
from crypto_steering.validation.tasks.shared.results import task_failure
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.contracts import _family_argument_contract_failure
from crypto_steering.validation.tasks.f05.api_family import Analysis
from crypto_steering.validation.tasks.shared.python_dataflow import (
    bytes_passed_to_encoded_parameter,
    source_proven_bytes_encoded_as_text,
)


def validate_python_encoding(
    context: TaskContext,
    analysis: Analysis,
    result: TaskValidationResult,
) -> TaskValidationResult | None:
    if context.language != "python":
        return None
    encoded_bytes = source_proven_bytes_encoded_as_text(context.source)
    if encoded_bytes is not None:
        return task_failure(
            result,
            "target_argument_contract_invalid",
            f"{encoded_bytes} is already bytes and cannot be encoded as text",
        )
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
    if context.language == "java":
        return None
    reason = _family_argument_contract_failure(
        context.language, context.source, family="F5"
    )
    if reason is None:
        return None
    return task_failure(result, "target_argument_contract_invalid", reason)
