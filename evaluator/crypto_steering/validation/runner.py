"""One validation engine for datasets, probes, and generated programs."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import signal
from typing import Iterable, Iterator, Sequence

from crypto_steering.config.models import ValidationConfig
from crypto_steering.validation.frontends import (
    FrontendRuntimeConfig,
    ToolchainRuntimeConfig,
    validate_frontends,
)
from crypto_steering.validation.internal import (
    ProgramAnalysis,
    analyze_program,
    skipped_program_analysis,
)
from crypto_steering.validation.models import (
    CompileOutcome,
    ImportOutcome,
    ProgramInput,
    ProviderExecution,
    ValidationResult,
)
from crypto_steering.validation.rules.aggregate import (
    program_rule_assessments,
    task_result_with_rule_assessments,
)


def validate_programs(
    programs: Sequence[ProgramInput],
    config: ValidationConfig,
    *,
    work_dir: str | Path,
    workers: int | None = None,
) -> tuple[ValidationResult, ...]:
    return tuple(
        iter_validation_results(
            programs,
            config,
            work_dir=work_dir,
            workers=workers,
        )
    )


def iter_validation_results(
    programs: Sequence[ProgramInput],
    config: ValidationConfig,
    *,
    work_dir: str | Path,
    workers: int | None = None,
) -> Iterator[ValidationResult]:
    ordered = tuple(programs)
    ids = [program.program_id for program in ordered]
    if not ordered or len(ids) != len(set(ids)):
        raise ValueError("Validation requires unique, non-empty program identities")
    root = Path(work_dir)
    frontend_audit = validate_frontends(
        _frontend_runtime_config(config),
        ordered,
        work_dir=root / "frontends",
        logs_dir=root / "logs",
    )
    compile_by_program = getattr(frontend_audit, "compile_by_program")
    worker_count = workers if workers is not None else config.execution.workers
    if worker_count < 1:
        raise ValueError("Validation worker count must be positive")
    payloads = (
        (
            program,
            config.policy.modern_guidance_iterations,
            config.execution.analysis_timeout_seconds,
            compile_by_program[program.program_id].parse_success,
        )
        for program in ordered
    )
    if worker_count == 1:
        analyses: Iterable[ProgramAnalysis] = (
            _analyze_worker(payload) for payload in payloads
        )
        yield from _join_results(ordered, analyses, frontend_audit)
        return
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        analyses = executor.map(_analyze_worker, payloads, chunksize=16)
        yield from _join_results(ordered, analyses, frontend_audit)


def _analyze_worker(payload: tuple[ProgramInput, int, int, bool]) -> ProgramAnalysis:
    program, modern_kdf_iterations, timeout_seconds, parse_success = payload
    if not parse_success:
        return skipped_program_analysis(
            program,
            "semantic analysis was skipped because the language parser rejected the source",
        )
    try:
        with _analysis_timeout(timeout_seconds):
            return analyze_program(
                program,
                modern_kdf_iterations=modern_kdf_iterations,
            )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        detector_errors = {
            misuse_id: ProviderExecution.error(message)
            for misuse_id in tuple(f"M{index:02d}" for index in range(1, 11))
        }
        task_keys = (
            tuple(detector_errors)
            if program.validation_mode == "multi_operation"
            else (program.target_misuse_id,)
            if program.target_misuse_id is not None
            else ()
        )
        return ProgramAnalysis(
            tasks={key: ProviderExecution.error(message) for key in task_keys},
            detectors=detector_errors,
        )


@contextmanager
def _analysis_timeout(seconds: int):
    """Bound one worker analysis on POSIX without changing scientific labels."""

    if seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return
    previous = signal.getsignal(signal.SIGALRM)

    def raise_timeout(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"program analysis exceeded {seconds} seconds")

    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _join_results(
    programs: Sequence[ProgramInput],
    analyses: Iterable[ProgramAnalysis],
    frontend_audit: object,
) -> Iterator[ValidationResult]:
    compile_by_program = getattr(frontend_audit, "compile_by_program")
    imports_by_program = getattr(frontend_audit, "imports_by_program")
    for program, analysis in zip(programs, analyses, strict=True):
        frontend = _frontend_execution(compile_by_program[program.program_id])
        parse_failed = bool(
            frontend.status == "completed"
            and frontend.value is not None
            and not frontend.value.parse_success
        )
        imports = (
            ProviderExecution.skipped(
                "import analysis was skipped because the language parser rejected the source"
            )
            if parse_failed
            else _import_execution(imports_by_program[program.program_id])
        )
        tasks = {
            key: (
                ProviderExecution.completed(
                    task_result_with_rule_assessments(execution.value)
                )
                if execution.status == "completed" and execution.value is not None
                else execution
            )
            for key, execution in analysis.tasks.items()
        }
        label_matches = {}
        for misuse_id, intended in program.intended_labels.items():
            execution = analysis.detectors.get(misuse_id)
            label_matches[misuse_id] = bool(
                execution is not None
                and execution.status == "completed"
                and execution.value is not None
                and execution.value.label == intended
            )
        completed_tasks = tuple(
            execution.value
            for execution in tasks.values()
            if execution.status == "completed" and execution.value is not None
        )
        task_providers_complete = bool(completed_tasks) and len(
            completed_tasks
        ) == len(tasks)
        required_tasks_complete = task_providers_complete and all(
            task.success for task in completed_tasks
        )
        validity_rules = program_rule_assessments(
            frontend.value if frontend.status == "completed" else None,
            imports.value if imports.status == "completed" else None,
            completed_tasks,
            task_providers_complete=task_providers_complete,
        )
        program_valid = all(rule.success for rule in validity_rules)
        reasons = []
        if frontend.status != "completed":
            reasons.append("frontend_provider_error")
        elif frontend.value is not None and not frontend.value.success:
            reasons.append(frontend.value.reason_code)
        if imports.status == "skipped":
            reasons.append("import_check_skipped_after_parse_failure")
        elif imports.status != "completed":
            reasons.append("import_provider_error")
        elif imports.value is not None and not imports.value.success:
            reasons.append(imports.value.reason_code)
        task_statuses = {execution.status for execution in tasks.values()}
        if "skipped" in task_statuses:
            reasons.append("target_task_skipped_after_parse_failure")
        elif not required_tasks_complete:
            reasons.append("target_task_invalid")
        if any(
            execution.status in {"unsupported", "error"}
            for execution in analysis.detectors.values()
        ):
            reasons.append("detector_provider_error")
        intended_not_evaluated = any(
            analysis.detectors.get(misuse_id) is not None
            and analysis.detectors[misuse_id].status == "skipped"
            for misuse_id in program.intended_labels
        )
        if intended_not_evaluated:
            reasons.append("intended_label_not_evaluated")
        elif any(not matched for matched in label_matches.values()):
            reasons.append("intended_label_mismatch")
        yield ValidationResult(
            program=program,
            frontend=frontend,
            imports=imports,
            tasks=tasks,
            detectors=analysis.detectors,
            program_valid=program_valid,
            intended_label_matches=label_matches,
            reason_codes=tuple(dict.fromkeys(reasons)),
            validity_rules=validity_rules,
        )


def _frontend_execution(
    outcome: CompileOutcome,
) -> ProviderExecution[CompileOutcome]:
    if outcome.reason_code == "analysis_error":
        return ProviderExecution.error(outcome.stderr or "frontend analysis error")
    return ProviderExecution.completed(outcome)


def _import_execution(outcome: ImportOutcome) -> ProviderExecution[ImportOutcome]:
    if outcome.reason_code == "analysis_error":
        return ProviderExecution.error("import analysis error")
    return ProviderExecution.completed(outcome)


def _frontend_runtime_config(config: ValidationConfig) -> FrontendRuntimeConfig:
    return FrontendRuntimeConfig(
        version=config.config_id,
        source_config_path="../scripts/config/evaluation.json",
        toolchains={
            frontend.language: ToolchainRuntimeConfig(frontend.raw)
            for frontend in config.frontends
        },
    )
