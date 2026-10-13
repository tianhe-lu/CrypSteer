"""Typed task-validator registry and probe operation mapping."""

from __future__ import annotations

from crypto_steering.validation.models import (
    DetectionResult,
    Language,
    ProviderExecution,
    TaskValidationResult,
    ValidationMode,
)
from crypto_steering.validation.source.index import SourceAnalyzer
from crypto_steering.validation.tasks.f01 import validate as validate_f01
from crypto_steering.validation.tasks.f02 import validate as validate_f02
from crypto_steering.validation.tasks.f03 import validate as validate_f03
from crypto_steering.validation.tasks.f04 import validate as validate_f04
from crypto_steering.validation.tasks.f05 import validate as validate_f05
from crypto_steering.validation.tasks.f06 import validate as validate_f06
from crypto_steering.validation.tasks.f07 import validate as validate_f07
from crypto_steering.validation.tasks.f08 import validate as validate_f08
from crypto_steering.validation.tasks.f09 import validate as validate_f09


FAMILY_VALIDATORS = {
    "F1": validate_f01,
    "F2": validate_f02,
    "F3": validate_f03,
    "F4": validate_f04,
    "F5": validate_f05,
    "F6": validate_f06,
    "F7": validate_f07,
    "F8": validate_f08,
    "F9": validate_f09,
}


CANONICAL_PROBE_TASKS: dict[str, str] = {
    # T suffixes preserve the original probe provenance. Family selection is
    # controlled by the F prefix, and every suffix uses the same rule set.
    "M01": "F1_T2",
    "M02": "F1_T2",
    "M03": "F2_T2",
    "M04": "F4_T1",
    "M05": "F9_T1",
    "M06": "F7_T1",
    "M07": "F7_T1",
    "M08": "F7_T1",
    "M09": "F5_T1",
    "M10": "F6_T1",
}


def validate_task(
    language: Language,
    source: str,
    *,
    task_id: str,
    target_misuse_id: str,
    mode: ValidationMode = "strict",
    target_detection: DetectionResult | None = None,
    indexed_analyzer: SourceAnalyzer | None = None,
) -> ProviderExecution[TaskValidationResult]:
    from crypto_steering.validation.tasks.engine import (
        validate_target_task,
        validate_target_task_scaffold,
    )

    try:
        if mode == "original_scaffold":
            if target_detection is None:
                raise ValueError(
                    "original-scaffold validation requires the completed target detector"
                )
            result = validate_target_task_scaffold(
                language,
                source,
                task_id=task_id,
                target_misuse_id=target_misuse_id,
                target_detection=target_detection,
                indexed_analyzer=indexed_analyzer,
                allow_independent_routes=True,
            )
        else:
            result = validate_target_task(
                language,
                source,
                task_id=task_id,
                target_misuse_id=target_misuse_id,
                target_detection=target_detection,
                indexed_analyzer=indexed_analyzer,
                allow_independent_routes=mode != "multi_operation",
            )
    except Exception as exc:
        return ProviderExecution.error(f"{type(exc).__name__}: {exc}")
    return ProviderExecution.completed(result)


def validate_probe_operations(
    language: Language,
    source: str,
    detections: dict[str, DetectionResult],
    *,
    indexed_analyzer: SourceAnalyzer | None = None,
) -> dict[str, ProviderExecution[TaskValidationResult]]:
    """Validate each independently declared composition operation."""

    return {
        misuse_id: validate_task(
            language,
            source,
            task_id=task_id,
            target_misuse_id=misuse_id,
            mode="multi_operation",
            target_detection=detections.get(misuse_id),
            indexed_analyzer=indexed_analyzer,
        )
        for misuse_id, task_id in CANONICAL_PROBE_TASKS.items()
    }
