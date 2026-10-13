"""F6 Rule 4: require MAC input and terminal operations."""

from __future__ import annotations

from crypto_steering.validation.java.task_analysis import _receiver_method_calls
from crypto_steering.validation.models import TaskValidationResult
from crypto_steering.validation.tasks.shared.results import java_task_result, require_analyzer
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.f06.api_family import Analysis


def validate_rule(
    context: TaskContext, analysis: Analysis
) -> TaskValidationResult | None:
    if context.language == "java":
        if not analysis.candidate_initializations:
            return java_task_result(
                context,
                success=False,
                failure_code="target_operation_incomplete",
                reason="F6 requires Mac.init with key material",
            )
        if not _receiver_method_calls(analysis.source, "doFinal"):
            return java_task_result(
                context,
                success=False,
                failure_code="target_operation_incomplete",
                reason="F6 requires Mac.doFinal after valid Mac initialization",
            )
        return None
    result = indexed_result(context, analysis)
    return None if result.success else result


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
    return java_task_result(
        context,
        success=True,
        reason="matched correct F6 Mac initialization and doFinal usage",
        matched_symbols=("Mac",),
    )
