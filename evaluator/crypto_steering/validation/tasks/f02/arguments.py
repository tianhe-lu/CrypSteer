"""F2 Rule 5: validate CBC initialization argument roles."""

from __future__ import annotations

from dataclasses import dataclass

from crypto_steering.validation.java.task_analysis import (
    _is_cipher_mode_expression,
    _is_iv_parameter_spec_expression,
    _is_key_expression,
)
from crypto_steering.validation.models import TaskValidationResult
from crypto_steering.validation.tasks.shared.results import java_task_result, task_failure
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.contracts import (
    _family_argument_contract_failure,
    _java_symmetric_argument_contract_failure,
)
from crypto_steering.validation.tasks.f02.api_family import Analysis, required_evidence


@dataclass(frozen=True)
class Initialization:
    receivers: frozenset[str]
    matched_iv_spec: bool


def resolve_initialization(analysis: Analysis) -> Initialization:
    receivers: set[str] = set()
    matched_iv_spec = False
    for receiver, call in analysis.candidate_initializations:
        if len(call.arguments) < 2:
            continue
        mode = call.arguments[0].text
        if not _is_cipher_mode_expression(analysis.source, mode, call.start):
            continue
        if not _is_key_expression(
            analysis.source, call.arguments[1].text, call.start
        ):
            continue
        if len(call.arguments) == 2 and "Cipher.ENCRYPT_MODE" in mode:
            receivers.add(receiver)
        if len(call.arguments) > 2 and _is_iv_parameter_spec_expression(
            analysis.source, call.arguments[2].text, call.start
        ):
            receivers.add(receiver)
            matched_iv_spec = True
    return Initialization(frozenset(receivers), matched_iv_spec)


def validate_rule(
    context: TaskContext, analysis: Analysis
) -> TaskValidationResult | None:
    if context.language == "java":
        if (
            analysis.candidate_initializations
            and not resolve_initialization(analysis).receivers
        ):
            return java_task_result(
                context,
                success=False,
                failure_code="target_argument_contract_invalid",
                reason=(
                    "F2 requires CBC Cipher.init with ENCRYPT/DECRYPT_MODE, key "
                    "material, and a valid IV pattern"
                ),
            )
        reason = _java_symmetric_argument_contract_failure(
            context.source,
            family="F2",
        )
        if reason is not None:
            return java_task_result(
                context,
                success=False,
                failure_code="target_argument_contract_invalid",
                reason=reason,
            )
        return None
    if any(item is None for item, _ in required_evidence(analysis)):
        return None
    reason = _family_argument_failure(context, analysis)
    if reason is None:
        return None
    from crypto_steering.validation.tasks.f02.operations import success_result

    return task_failure(
        success_result(context, analysis),
        "target_argument_contract_invalid",
        reason,
    )


def _family_argument_failure(
    context: TaskContext, analysis: Analysis
) -> str | None:
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
        family="F2",
        directions=directions,
    )
