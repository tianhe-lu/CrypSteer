"""F1 Rule 5: validate structural argument roles and positions."""

from __future__ import annotations

from crypto_steering.validation.java.task_analysis import (
    _is_cipher_mode_expression,
    _is_key_expression,
)
from crypto_steering.validation.models import TaskValidationResult
from crypto_steering.validation.tasks.shared.results import java_task_result, task_failure
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.contracts import (
    _family_argument_contract_failure,
    _java_symmetric_argument_contract_failure,
)
from crypto_steering.validation.tasks.f01.api_family import Analysis
from crypto_steering.validation.tasks.f01.api_family import required_evidence
from crypto_steering.validation.tasks.shared.python_dataflow import (
    bytes_passed_to_encoded_parameter,
    text_passed_to_binary_parameter,
)


def initialized_receivers(analysis: Analysis) -> frozenset[str]:
    receivers: set[str] = set()
    for receiver, call in analysis.candidate_initializations:
        if len(call.arguments) < 2:
            continue
        if not _is_cipher_mode_expression(
            analysis.source, call.arguments[0].text, call.start
        ):
            continue
        if _is_key_expression(analysis.source, call.arguments[1].text, call.start):
            receivers.add(receiver)
    return frozenset(receivers)


def validate_rule(
    context: TaskContext, analysis: Analysis
) -> TaskValidationResult | None:
    if context.language == "java":
        if analysis.candidate_initializations and not initialized_receivers(analysis):
            return java_task_result(
                context,
                success=False,
                failure_code="target_argument_contract_invalid",
                reason="F1 requires Cipher.init with ENCRYPT/DECRYPT_MODE and key material",
            )
        reason = _java_symmetric_argument_contract_failure(
            context.source,
            family="F1",
        )
        if reason is not None:
            return java_task_result(
                context,
                success=False,
                failure_code="target_argument_contract_invalid",
                reason=reason,
            )
        return None

    if any(
        item is None for item, _ in required_evidence(analysis)
    ):
        return None
    from crypto_steering.validation.tasks.f01.operations import indexed_result

    base = indexed_result(context, analysis)
    if context.language == "python":
        conflict = bytes_passed_to_encoded_parameter(context.source)
        if conflict is not None:
            function, parameter = conflict
            return task_failure(
                base,
                "target_argument_contract_invalid",
                f"{function} receives bytes for {parameter}, then calls str.encode on it",
            )
        conflict = text_passed_to_binary_parameter(context.source)
        if conflict is not None:
            function, parameter = conflict
            return task_failure(
                base,
                "target_argument_contract_invalid",
                f"{function} receives text for {parameter}, then passes it to a bytes-only operation",
            )
    reason = _family_argument_failure(context, analysis)
    if reason is None:
        return None
    return task_failure(base, "target_argument_contract_invalid", reason)


def _family_argument_failure(
    context: TaskContext, analysis: Analysis
) -> str | None:
    """Accept any observed direction with structurally usable arguments."""

    directions = frozenset(
        direction
        for direction, evidence in (
            ("encrypt", analysis.encrypt),
            ("decrypt", analysis.decrypt),
        )
        if evidence is not None
    )
    return _family_argument_contract_failure(
        context.language,
        context.source,
        family="F1",
        directions=directions,
    )
