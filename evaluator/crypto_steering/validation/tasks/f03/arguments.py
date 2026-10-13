"""F3 Rule 5: validate RSA initialization argument roles."""

from __future__ import annotations

import re

from crypto_steering.validation.java.task_analysis import (
    _is_asymmetric_key_expression,
)
from crypto_steering.validation.models import TaskValidationResult
from crypto_steering.validation.tasks.shared.results import java_task_result, task_failure
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.contracts import (
    _family_argument_contract_failure,
)
from crypto_steering.validation.tasks.f03.api_family import Analysis
from crypto_steering.validation.tasks.f03.operations import success_result


def initialized_receivers(analysis: Analysis) -> frozenset[str]:
    receivers: set[str] = set()
    for receiver, call in analysis.candidate_initializations:
        if len(call.arguments) < 2:
            continue
        if re.search(
            r"\bCipher\.(?:ENCRYPT|DECRYPT)_MODE\b", call.arguments[0].text
        ) is None:
            continue
        if _is_asymmetric_key_expression(
            analysis.source, call.arguments[1].text, call.start
        ):
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
                reason=(
                    "F3 requires RSA Cipher.init with ENCRYPT/DECRYPT_MODE and "
                    "public/private key material"
                ),
            )
        return None
    from crypto_steering.validation.tasks.f03.api_family import required_evidence

    if any(item is None for item, _ in required_evidence(analysis)):
        return None
    reason = _family_argument_contract_failure(
        context.language,
        context.source,
        family="F3",
        directions=frozenset(
            direction
            for direction, evidence in (
                ("encrypt", analysis.encrypt),
                ("decrypt", analysis.decrypt),
            )
            if evidence is not None
        ),
    )
    if reason is None:
        return None
    return task_failure(
        success_result(context, analysis),
        "target_argument_contract_invalid",
        reason,
    )
