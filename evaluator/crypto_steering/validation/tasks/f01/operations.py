"""F1 Rule 4: require one completed symmetric operation."""

from __future__ import annotations

from crypto_steering.validation.java.task_analysis import _receiver_method_calls
from crypto_steering.validation.models import TaskValidationResult
from crypto_steering.validation.tasks.shared.results import java_task_result, require_analyzer
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.f01.api_family import Analysis, required_evidence


def validate_rule(
    context: TaskContext, analysis: Analysis
) -> TaskValidationResult | None:
    if context.language == "java":
        if not analysis.candidate_initializations:
            return java_task_result(
                context,
                success=False,
                failure_code="target_operation_incomplete",
                reason="F1 requires Cipher.init with ENCRYPT/DECRYPT_MODE and key material",
            )
        if not _receiver_method_calls(analysis.source, "doFinal"):
            return java_task_result(
                context,
                success=False,
                failure_code="target_operation_incomplete",
                reason="F1 requires Cipher.doFinal after valid Cipher initialization",
            )
        return None

    required = required_evidence(analysis)
    missing = tuple(name for item, name in required if item is None)
    if not missing:
        return None
    analyzer = require_analyzer(context)
    return TaskValidationResult(
        language=analyzer.language,
        task_id=context.task_id,
        success=False,
        reason_codes=("target_operation_incomplete",),
        reason="missing " + ", ".join(missing),
        matched_operations=tuple(item for item, _ in required if item is not None),
    )


def success_result(context: TaskContext, analysis: Analysis) -> TaskValidationResult:
    if context.language == "java":
        return java_task_result(
            context,
            success=True,
            reason="matched correct F1 Cipher initialization and doFinal usage",
            matched_symbols=("Cipher",),
        )
    analyzer = require_analyzer(context)
    return TaskValidationResult(
        language=analyzer.language,
        task_id=context.task_id,
        success=True,
        reason_codes=(),
        reason="completed symmetric encryption or decryption operation",
        matched_operations=tuple(
            item for item, _ in required_evidence(analysis)
            if item is not None
        ),
    )


def indexed_result(context: TaskContext, analysis: Analysis) -> TaskValidationResult:
    """Build the legacy indexed result before Rules 5 and 6 refine it."""

    required = required_evidence(analysis)
    missing = tuple(name for item, name in required if item is None)
    analyzer = require_analyzer(context)
    matched = tuple(item for item, _ in required if item is not None)
    if missing:
        return TaskValidationResult(
            language=analyzer.language,
            task_id=context.task_id,
            success=False,
            reason_codes=(
                "target_api_not_found" if not matched else "target_operation_incomplete",
            ),
            reason="missing " + ", ".join(missing),
            matched_operations=matched,
        )
    return success_result(context, analysis)
