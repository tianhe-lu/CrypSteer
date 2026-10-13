"""Internal provider orchestration over one indexed program."""

from __future__ import annotations

from dataclasses import dataclass

from crypto_steering.validation.detectors.registry import detect_all, detect_misuse
from crypto_steering.validation.models import (
    DetectionResult,
    MISUSE_IDS,
    ProgramInput,
    ProviderExecution,
    TaskValidationResult,
)
from crypto_steering.validation.operation_scopes import (
    scope_detector_executions_to_operations,
)
from crypto_steering.validation.source.models import IndexedProgram, index_program
from crypto_steering.validation.tasks.registry import (
    CANONICAL_PROBE_TASKS,
    validate_probe_operations,
    validate_task,
)


@dataclass(frozen=True)
class ProgramAnalysis:
    tasks: dict[str, ProviderExecution[TaskValidationResult]]
    detectors: dict[str, ProviderExecution[DetectionResult]]


def skipped_program_analysis(
    program: ProgramInput,
    message: str,
) -> ProgramAnalysis:
    """Record that semantic providers were not run after parsing failed."""

    detector_executions = {
        misuse_id: ProviderExecution.skipped(message) for misuse_id in MISUSE_IDS
    }
    task_keys = (
        MISUSE_IDS
        if program.validation_mode == "multi_operation"
        else (program.target_misuse_id,)
        if program.target_misuse_id is not None
        else ()
    )
    return ProgramAnalysis(
        tasks={key: ProviderExecution.skipped(message) for key in task_keys},
        detectors=detector_executions,
    )


def analyze_program(
    program: ProgramInput,
    *,
    modern_kdf_iterations: int = 100_000,
) -> ProgramAnalysis:
    try:
        indexed = index_program(program)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        errors = {
            misuse_id: ProviderExecution.error(message) for misuse_id in MISUSE_IDS
        }
        return ProgramAnalysis(tasks={}, detectors=errors)

    if program.validation_mode == "multi_operation":
        detectors = {
            misuse_id: detect_misuse(
                program.language,
                misuse_id,
                program.source,
                task_id=task_id,
                modern_kdf_iterations=modern_kdf_iterations,
                analysis_profile=program.analysis_profile,
                indexed_analyzer=indexed.analyzer,
            )
            for misuse_id, task_id in CANONICAL_PROBE_TASKS.items()
        }
        detectors = scope_detector_executions_to_operations(
            program.language,
            program.source,
            detectors,
            program.operation_functions,
            indexed_analyzer=indexed.analyzer,
        )
    else:
        detectors = detect_all(
            program.language,
            program.source,
            task_id=program.task_id,
            modern_kdf_iterations=modern_kdf_iterations,
            analysis_profile=program.analysis_profile,
            indexed_analyzer=indexed.analyzer,
        )
    completed = {
        misuse_id: execution.value
        for misuse_id, execution in detectors.items()
        if execution.status == "completed" and execution.value is not None
    }
    if program.validation_mode == "multi_operation":
        tasks = validate_probe_operations(
            program.language,
            program.source,
            completed,
            indexed_analyzer=indexed.analyzer,
        )
    elif program.task_id is not None and program.target_misuse_id is not None:
        tasks = {
            program.target_misuse_id: validate_task(
                program.language,
                program.source,
                task_id=program.task_id,
                target_misuse_id=program.target_misuse_id,
                mode=program.validation_mode,
                target_detection=completed.get(program.target_misuse_id),
                indexed_analyzer=indexed.analyzer,
            )
        }
    else:
        tasks = {}
    return ProgramAnalysis(tasks=tasks, detectors=detectors)


class UnsupportedValidationProvider:
    """Honest placeholder that never emits a scientific label."""

    provider_id = "third_party"

    @staticmethod
    def detect(*_args: object, **_kwargs: object) -> ProviderExecution[DetectionResult]:
        return ProviderExecution.unsupported("third-party misuse detection is not implemented")

    @staticmethod
    def validate(*_args: object, **_kwargs: object) -> ProviderExecution[TaskValidationResult]:
        return ProviderExecution.unsupported("third-party task validation is not implemented")
