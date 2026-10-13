"""F5 Rule 4: require input and completed digest production."""

from __future__ import annotations

from crypto_steering.validation.models import TaskValidationResult
from crypto_steering.validation.tasks.shared.results import java_task_result, require_analyzer
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.f05.api_family import Analysis


def validate_rule(
    context: TaskContext, analysis: Analysis
) -> TaskValidationResult | None:
    if context.language == "java":
        if not (analysis.chained_digest or analysis.completed_digest):
            if analysis.has_digest_terminal:
                return None
            return java_task_result(
                context,
                success=False,
                failure_code="target_operation_incomplete",
                reason="F5 requires MessageDigest.digest on input data or after update",
            )
        result = success_result(context, analysis)
    else:
        result = indexed_result(context, analysis)
        if not result.success:
            return result
    return None


def indexed_result(context: TaskContext, analysis: Analysis) -> TaskValidationResult:
    analyzer = require_analyzer(context)
    missing = tuple(name for item, name in analysis.required if item is None)
    matched = tuple(item for item, _ in analysis.required if item is not None)
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
    if analysis.chained_digest:
        reason = "matched chained F5 MessageDigest.getInstance(...).digest usage"
    else:
        reason = "matched correct F5 MessageDigest.digest usage"
    return java_task_result(
        context,
        success=True,
        reason=reason,
        matched_symbols=("MessageDigest",),
    )
