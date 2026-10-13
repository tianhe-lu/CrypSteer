"""F8 Rule 4: require protected serialization and storage operations."""

from __future__ import annotations

from crypto_steering.validation.models import TaskValidationResult
from crypto_steering.validation.tasks.shared.results import java_task_result, require_analyzer
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.f08.api_family import Analysis


def validate_presence(
    context: TaskContext, analysis: Analysis
) -> TaskValidationResult | None:
    if context.language != "java" or analysis.keystore_operations:
        return None
    return java_task_result(
        context,
        success=False,
        failure_code="target_operation_incomplete",
        reason="F8 requires actual KeyStore load/store/set/get operation usage",
    )


def validate_rule(
    context: TaskContext, analysis: Analysis
) -> TaskValidationResult | None:
    if context.language == "java":
        # Rule 4 presence was established by ``validate_presence``. The
        # material subtype encoded by T1/T2/T3 is provenance, not policy.
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
    matched = tuple(
        sorted({call.name for call in analysis.keystore_operations} | {"KeyStore"})
    )
    return java_task_result(
        context,
        success=True,
        reason="matched correct F8 KeyStore operation usage",
        matched_symbols=matched,
    )
