"""F7 Rule 4: require a completed password-key derivation."""

from __future__ import annotations

from crypto_steering.validation.models import TaskValidationResult
from crypto_steering.validation.tasks.shared.results import java_task_result, require_analyzer
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.f07.api_family import Analysis


def validate_rule(
    context: TaskContext, analysis: Analysis
) -> TaskValidationResult | None:
    if context.language == "java":
        if analysis.valid_generate_secret:
            return None
        return java_task_result(
            context,
            success=False,
            failure_code="target_operation_incomplete",
            reason=(
                "F7 requires generateSecret with a PBEKeySpec/KeySpec derived "
                "from PBEKeySpec"
            ),
        )
    if analysis.terminal is not None:
        return None
    return TaskValidationResult(
        language=context.language,
        task_id=context.task_id,
        success=False,
        reason_codes=("target_operation_incomplete",),
        reason="missing completed PBKDF2 derivation or verification",
        matched_operations=tuple(
            item for item in (analysis.setup,) if item is not None
        ),
    )


def indexed_result(context: TaskContext, analysis: Analysis) -> TaskValidationResult:
    analyzer = require_analyzer(context)
    required = (
        (analysis.setup, "PBKDF2 construction"),
        (analysis.argument_contract, "usable PBKDF2 arguments"),
        (analysis.terminal, "key derivation terminal operation"),
    )
    missing = tuple(name for item, name in required if item is None)
    matched = tuple(item for item, _ in required if item is not None)
    return TaskValidationResult(
        language=analyzer.language,
        task_id=context.task_id,
        success=not missing,
        reason_codes=(
            () if not missing else (
                "target_api_not_found" if not matched else "target_operation_incomplete",
            )
        ),
        reason=(
            "completed target-task operation"
            if not missing else "missing " + ", ".join(missing)
        ),
        matched_operations=matched,
    )


def success_result(context: TaskContext, analysis: Analysis) -> TaskValidationResult:
    if context.language != "java":
        return indexed_result(context, analysis)
    return java_task_result(
        context,
        success=True,
        reason="matched correct F7 PBEKeySpec and SecretKeyFactory usage",
        matched_symbols=("PBEKeySpec", "SecretKeyFactory"),
    )
