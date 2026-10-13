"""Task-neutral result helpers shared by task-family validators."""

from __future__ import annotations

from crypto_steering.validation.models import (
    DetectionResult,
    TaskEvidence,
    TaskValidationResult,
)
from crypto_steering.validation.tasks.shared.context import TaskContext


def require_analyzer(context: TaskContext):
    if context.analyzer is None:
        raise ValueError(f"{context.language} task validation requires a source analyzer")
    return context.analyzer


def task_failure(
    result: TaskValidationResult,
    reason_code: str,
    reason: str,
) -> TaskValidationResult:
    return TaskValidationResult(
        language=result.language,
        task_id=result.task_id,
        success=False,
        reason_codes=(reason_code,),
        reason=reason,
        matched_operations=result.matched_operations,
    )


def java_task_result(
    context: TaskContext,
    *,
    success: bool,
    reason: str,
    failure_code: str | None = None,
    matched_symbols: tuple[str, ...] = (),
) -> TaskValidationResult:
    if success:
        reason_codes: tuple[str, ...] = ()
    else:
        if failure_code is None:
            raise ValueError("a failed Java task assessment requires a reason code")
        reason_codes = (failure_code,)
    return TaskValidationResult(
        language="java",
        task_id=context.task_id,
        success=success,
        reason_codes=reason_codes,
        reason=reason,
        matched_operations=tuple(
            TaskEvidence(operation=symbol, expression=symbol)
            for symbol in matched_symbols
        ),
    )


def target_detection(context: TaskContext) -> DetectionResult:
    if context.target_detection is not None:
        return context.target_detection
    from crypto_steering.validation.detectors.registry import detect_misuse

    execution = detect_misuse(
        context.language,
        context.target_misuse_id,
        context.source,
        task_id=context.task_id,
        indexed_analyzer=context.analyzer,
    )
    if execution.status != "completed" or execution.value is None:
        raise RuntimeError(
            execution.message
            or f"{context.target_misuse_id} detector did not complete"
        )
    return execution.value


def typescript_node_decryption_evidence(
    context: TaskContext,
    evidence: TaskEvidence | None,
) -> TaskEvidence | None:
    """Do not count a local method named decrypt as a Node decipher."""

    analyzer = require_analyzer(context)
    if analyzer.language != "typescript" or evidence is None:
        return evidence
    if analyzer.contains_call("createCipheriv") and not analyzer.contains_call(
        "createDecipheriv"
    ):
        return None
    return evidence
