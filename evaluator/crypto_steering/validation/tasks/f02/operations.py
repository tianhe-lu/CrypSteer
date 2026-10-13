"""F2 Rule 4: require one completed CBC operation."""

from __future__ import annotations

from crypto_steering.validation.java.task_analysis import _receiver_method_calls
from crypto_steering.validation.models import TaskValidationResult
from crypto_steering.validation.tasks.shared.results import java_task_result, require_analyzer
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.f02.api_family import Analysis, required_evidence
from crypto_steering.validation.tasks.f02.arguments import resolve_initialization


def validate_rule(
    context: TaskContext, analysis: Analysis
) -> TaskValidationResult | None:
    if context.language == "java":
        if not analysis.candidate_initializations:
            return java_task_result(
                context,
                success=False,
                failure_code="target_operation_incomplete",
                reason=(
                    "F2 requires CBC Cipher.init with ENCRYPT/DECRYPT_MODE, key "
                    "material, and a valid IV pattern"
                ),
            )
        if not _receiver_method_calls(analysis.source, "doFinal"):
            return java_task_result(
                context,
                success=False,
                failure_code="target_operation_incomplete",
                reason="F2 requires Cipher.doFinal after valid CBC initialization",
            )
        return None
    result = indexed_result(context, analysis)
    return None if result.success else result


def indexed_result(context: TaskContext, analysis: Analysis) -> TaskValidationResult:
    analyzer = require_analyzer(context)
    required = required_evidence(analysis)
    missing = tuple(name for item, name in required if item is None)
    matched = tuple(item for item, _ in required if item is not None)
    return TaskValidationResult(
        language=analyzer.language,
        task_id=context.task_id,
        success=not missing,
        reason_codes=(
            ()
            if not missing
            else ("target_api_not_found" if not matched else "target_operation_incomplete",)
        ),
        reason=(
            "completed target-task operation"
            if not missing
            else "missing " + ", ".join(missing)
        ),
        matched_operations=matched,
    )


def success_result(context: TaskContext, analysis: Analysis) -> TaskValidationResult:
    if context.language != "java":
        return indexed_result(context, analysis)
    initialization = resolve_initialization(analysis)
    matched = (
        ("Cipher", "CBC", "IvParameterSpec")
        if initialization.matched_iv_spec
        else ("Cipher", "CBC")
    )
    return java_task_result(
        context,
        success=True,
        reason="matched correct F2 CBC initialization and doFinal usage",
        matched_symbols=matched,
    )
