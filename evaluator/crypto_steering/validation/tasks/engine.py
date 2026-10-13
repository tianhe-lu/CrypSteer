"""Public orchestration for structural F1-F9 task validation."""

from __future__ import annotations

from crypto_steering.validation.models import (
    DetectionResult,
    Language,
    TaskValidationResult,
)
from crypto_steering.validation.rules.aggregate import (
    task_result_with_rule_assessments,
)
from crypto_steering.validation.source.index import SourceAnalyzer, analyzer_for
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.shared.routes import validate_any_route
from crypto_steering.validation.tasks.contracts import (
    _native_scaffold_completion_contract,
)
from crypto_steering.validation.tasks.f01 import mode_sink_is_rewriteable
from crypto_steering.validation.tasks.f09.program_contract import python_random_program_failure
from crypto_steering.validation.tasks.shared.results import task_failure
from crypto_steering.validation.tasks.registry import FAMILY_VALIDATORS


def validate_target_task(
    language: Language,
    source: str,
    *,
    task_id: str,
    target_misuse_id: str,
    target_detection: DetectionResult | None = None,
    indexed_analyzer: SourceAnalyzer | None = None,
    allow_independent_routes: bool = True,
) -> TaskValidationResult:
    """Validate API-family use, operations, arguments, and sequence."""

    family = task_id.split("_", 1)[0].upper()
    validator = FAMILY_VALIDATORS.get(family)
    if validator is None:
        return task_result_with_rule_assessments(
            TaskValidationResult(
                language=language,
                task_id=task_id,
                success=False,
                reason_codes=("target_wrong_api_family",),
                reason=f"no target-task validator is configured for {family}",
            )
        )

    analyzer = indexed_analyzer
    if language != "java" and analyzer is None:
        try:
            analyzer = analyzer_for(language, source)
        except Exception as exc:
            return task_result_with_rule_assessments(
                TaskValidationResult(
                    language=language,
                    task_id=task_id,
                    success=False,
                    reason_codes=("syntax_error",),
                    reason=(
                        "source could not be indexed for target-task validation: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            )

    context = TaskContext(
        language=language,
        source=source,
        task_id=task_id,
        target_misuse_id=target_misuse_id,
        analyzer=analyzer,
        target_detection=target_detection,
        allow_independent_routes=allow_independent_routes,
    )
    result = validate_any_route(context, validator)
    if result.success and language == "python" and family == "F9":
        # A callable RNG helper must not hide a definite error on the actual
        # program entry path. Keep this after independent-route selection so
        # projecting out main cannot erase the failure.
        failure = python_random_program_failure(source)
        if failure is not None:
            return task_result_with_rule_assessments(
                task_failure(result, "target_argument_contract_invalid", failure),
                refresh=True,
            )
    return result


def validate_target_task_scaffold(
    language: Language,
    source: str,
    *,
    task_id: str,
    target_misuse_id: str,
    target_detection: DetectionResult,
    indexed_analyzer: SourceAnalyzer | None = None,
    allow_independent_routes: bool = True,
) -> TaskValidationResult:
    """Apply native-original curation checks after structural validation."""

    result = validate_target_task(
        language,
        source,
        task_id=task_id,
        target_misuse_id=target_misuse_id,
        target_detection=target_detection,
        indexed_analyzer=indexed_analyzer,
        allow_independent_routes=allow_independent_routes,
    )
    rewriteable_m02_sink = (
        target_misuse_id == "M02"
        and mode_sink_is_rewriteable(language, source)
    )
    if result.reason_codes == ("target_argument_unresolved",) or (
        rewriteable_m02_sink
        and result.reason_codes == ("target_argument_missing",)
    ):
        result = validate_target_task(
            language,
            source,
            task_id=task_id,
            target_misuse_id=target_misuse_id,
            target_detection=None,
            indexed_analyzer=indexed_analyzer,
            allow_independent_routes=allow_independent_routes,
        )
    if not result.success:
        return result
    if target_misuse_id == "M02" and not rewriteable_m02_sink:
        return task_result_with_rule_assessments(
            TaskValidationResult(
                language=language,
                task_id=task_id,
                success=False,
                reason_codes=("target_api_not_rewriteable",),
                reason=(
                    "M02 native originals require a mode-configurable symmetric "
                    "cipher API; a fixed AEAD primitive cannot be rewritten into "
                    "the controlled ECB/CBC mode contrast"
                ),
                matched_operations=result.matched_operations,
            )
        )
    native_failure = _native_scaffold_completion_contract(
        language,
        source,
        task_id,
        result,
    )
    return task_result_with_rule_assessments(
        native_failure or result,
        refresh=native_failure is not None,
    )
