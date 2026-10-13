"""F4 Rule 4: require one completed signing or verification route."""

from __future__ import annotations

from crypto_steering.validation.java.task_analysis import _receiver_method_calls
from crypto_steering.validation.models import TaskValidationResult
from crypto_steering.validation.tasks.shared.results import java_task_result, require_analyzer
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.f04.api_family import Analysis, required_evidence


def validate_rule(
    context: TaskContext, analysis: Analysis
) -> TaskValidationResult | None:
    if context.language == "java":
        completed_sign, completed_verify = completed_routes(analysis)
        if not (completed_sign or completed_verify):
            present = {
                name
                for name in ("initSign", "initVerify", "update", "sign", "verify")
                if _receiver_method_calls(analysis.source, name)
            }
            if present:
                return java_task_result(
                    context,
                    success=False,
                    failure_code="target_operation_incomplete",
                    reason=(
                        "F4 requires one connected init/update/sign or "
                        "init/update/verify route"
                    ),
                )
            return java_task_result(
                context,
                success=False,
                failure_code="target_api_not_found",
                reason="F4 requires a Signature sign or verify operation",
            )
        return None
    result = indexed_result(context, analysis)
    return None if result.success else result


def completed_routes(analysis: Analysis) -> tuple[bool, bool]:
    from crypto_steering.validation.java.task_analysis import (
        _is_asymmetric_key_expression,
    )

    completed_sign = False
    completed_verify = False
    for _, calls in analysis.calls_by_receiver:
        mode: str | None = None
        updated = False
        for _, method_name, call in sorted(calls):
            if method_name in {"initSign", "initVerify"}:
                if not call.arguments or not _is_asymmetric_key_expression(
                    analysis.source, call.arguments[0].text, call.start
                ):
                    mode = None
                    updated = False
                    continue
                mode = "sign" if method_name == "initSign" else "verify"
                updated = False
            elif method_name == "update" and mode is not None:
                updated = bool(call.arguments)
            elif method_name == "sign" and mode == "sign" and updated:
                completed_sign = True
            elif method_name == "verify" and mode == "verify" and updated:
                completed_verify = True
    return completed_sign, completed_verify


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
        reason="matched a complete F4 signing or verification route",
        matched_symbols=("Signature",),
    )
