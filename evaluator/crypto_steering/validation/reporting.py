"""Streaming, table-first validation artifacts and strict summaries."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Iterable

from crypto_steering.validation.models import ValidationResult, jsonable


class ValidationSummary:
    def __init__(self, payload: dict[str, object], path: Path) -> None:
        self.payload = payload
        self.path = path

    def to_dict(self) -> dict[str, object]:
        return dict(self.payload)


def write_validation_results(
    results: Iterable[ValidationResult],
    output_dir: str | Path,
    *,
    corpus_id: str,
    validation_config_id: str,
) -> ValidationSummary:
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Validation output directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "program": root / "program_results.jsonl",
        "frontend": root / "frontend_results.jsonl",
        "task": root / "task_results.jsonl",
        "detector": root / "detector_results.jsonl",
        "finding": root / "detector_findings.jsonl",
        "mismatch": root / "intended_label_mismatches.jsonl",
        "failure": root / "failures.jsonl",
    }
    handles = {
        key: path.open("w", encoding="utf-8", newline="\n")
        for key, path in paths.items()
    }
    counts: Counter[str] = Counter()
    by_corpus: dict[str, Counter[str]] = defaultdict(Counter)
    by_language: dict[str, Counter[str]] = defaultdict(Counter)
    by_validity_rule: dict[str, Counter[str]] = defaultdict(Counter)
    try:
        for result in results:
            program = result.program
            counts["programs"] += 1
            counts["program_valid"] += int(result.program_valid)
            counts["program_invalid"] += int(not result.program_valid)
            counts["intended_label_assertions"] += len(result.intended_label_matches)
            matches = sum(result.intended_label_matches.values())
            counts["intended_label_matches"] += matches
            counts["intended_label_mismatches"] += (
                len(result.intended_label_matches) - matches
            )
            counts["provider_errors"] += len(result.provider_errors)
            counts["frontend_success"] += int(
                result.frontend.status == "completed"
                and result.frontend.value is not None
                and result.frontend.value.success
            )
            counts["import_success"] += int(
                result.imports.status == "completed"
                and result.imports.value is not None
                and result.imports.value.success
            )
            counts["task_checks"] += len(result.tasks)
            counts["task_success"] += sum(
                execution.status == "completed"
                and execution.value is not None
                and execution.value.success
                for execution in result.tasks.values()
            )
            counts["task_skipped"] += sum(
                execution.status == "skipped"
                for execution in result.tasks.values()
            )
            counts["detector_checks"] += len(result.detectors)
            counts["detector_completed"] += sum(
                execution.status == "completed"
                for execution in result.detectors.values()
            )
            counts["detector_skipped"] += sum(
                execution.status == "skipped"
                for execution in result.detectors.values()
            )
            if not program.intended_labels:
                counts["expected_label_not_asserted"] += 1
            if program.target_misuse_id is not None:
                counts["unexpected_secondary_misuse"] += sum(
                    misuse_id != program.target_misuse_id
                    and execution.status == "completed"
                    and execution.value is not None
                    and execution.value.label in {"misuse", "mixed"}
                    for misuse_id, execution in result.detectors.items()
                )
            corpus_counts = by_corpus[program.corpus_kind]
            language_counts = by_language[program.language]
            for target in (corpus_counts, language_counts):
                target["programs"] += 1
                target["program_valid"] += int(result.program_valid)
                target["intended_label_assertions"] += len(result.intended_label_matches)
                target["intended_label_matches"] += matches
            for rule in result.validity_rules:
                by_validity_rule[rule.rule_id][rule.status] += 1

            _write(handles["program"], _program_row(result))
            _write(
                handles["frontend"],
                {
                    "program_id": program.program_id,
                    "frontend": _execution_row(result.frontend),
                    "imports": _execution_row(result.imports),
                },
            )
            for task_key, execution in result.tasks.items():
                _write(
                    handles["task"],
                    {
                        "program_id": program.program_id,
                        "task_key": task_key,
                        "execution": _execution_row(execution),
                    },
                )
            for misuse_id, execution in result.detectors.items():
                detector_row = {
                    "program_id": program.program_id,
                    "detector_misuse_id": misuse_id,
                    "operation_function": program.operation_functions.get(misuse_id),
                    "intended_label": program.intended_labels.get(misuse_id),
                    "execution": _execution_row(execution),
                }
                _write(handles["detector"], detector_row)
                if execution.status == "completed" and execution.value is not None:
                    for finding in execution.value.findings:
                        _write(
                            handles["finding"],
                            {
                                "program_id": program.program_id,
                                "detector_misuse_id": misuse_id,
                                "finding": jsonable(finding),
                            },
                        )
            for misuse_id, matched in result.intended_label_matches.items():
                if not matched:
                    execution = result.detectors[misuse_id]
                    _write(
                        handles["mismatch"],
                        {
                            "program_id": program.program_id,
                            "misuse_id": misuse_id,
                            "intended_label": program.intended_labels[misuse_id],
                            "observed_label": (
                                execution.value.label
                                if execution.status == "completed"
                                and execution.value is not None
                                else None
                            ),
                            "provider_status": execution.status,
                            "provider_message": execution.message,
                        },
                    )
            if not result.program_valid or result.provider_errors:
                _write(
                    handles["failure"],
                    {
                        "program_id": program.program_id,
                        "reason_codes": list(result.reason_codes),
                        "validity_rules": {
                            rule.rule_id: jsonable(rule)
                            for rule in result.validity_rules
                        },
                        "provider_errors": list(result.provider_errors),
                    },
                )
    finally:
        for handle in handles.values():
            handle.close()

    payload: dict[str, object] = {
        "schema_version": 1,
        "corpus_id": corpus_id,
        "validation_config_id": validation_config_id,
        **dict(sorted(counts.items())),
        "by_corpus_kind": {
            key: dict(sorted(value.items())) for key, value in sorted(by_corpus.items())
        },
        "by_language": {
            key: dict(sorted(value.items())) for key, value in sorted(by_language.items())
        },
        "by_validity_rule": {
            key: dict(sorted(value.items()))
            for key, value in sorted(by_validity_rule.items())
        },
    }
    summary_path = root / "summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ValidationSummary(payload, summary_path)


def _program_row(result: ValidationResult) -> dict[str, object]:
    program = result.program
    target_execution = (
        result.detectors.get(program.target_misuse_id)
        if program.target_misuse_id is not None
        else None
    )
    target_label = (
        target_execution.value.label
        if target_execution is not None
        and target_execution.status == "completed"
        and target_execution.value is not None
        else None
    )
    task_success = bool(result.tasks) and all(
        execution.status == "completed"
        and execution.value is not None
        and execution.value.success
        for execution in result.tasks.values()
    )
    secondary = tuple(
        misuse_id
        for misuse_id, execution in result.detectors.items()
        if misuse_id != program.target_misuse_id
        and execution.status == "completed"
        and execution.value is not None
        and execution.value.label in {"misuse", "mixed"}
    )
    return {
        "program_id": program.program_id,
        "relative_path": program.relative_path,
        "language": program.language,
        "corpus_kind": program.corpus_kind,
        "validation_mode": program.validation_mode,
        "analysis_profile": program.analysis_profile,
        "target_misuse_id": program.target_misuse_id,
        "task_id": program.task_id,
        "intended_labels": dict(program.intended_labels),
        "operation_functions": dict(program.operation_functions),
        "expected_rule": program.expected_rule,
        "expected_structure": program.expected_structure,
        "metadata": jsonable(program.metadata),
        "frontend_valid": bool(
            result.frontend.status == "completed"
            and result.frontend.value is not None
            and result.frontend.value.success
        ),
        "import_valid": bool(
            result.imports.status == "completed"
            and result.imports.value is not None
            and result.imports.value.success
        ),
        "target_task_valid": task_success,
        "target_detector_label": target_label,
        "secondary_misuse_ids": list(secondary),
        "program_valid": result.program_valid,
        "validity_rules": {
            rule.rule_id: jsonable(rule) for rule in result.validity_rules
        },
        "intended_label_matches": dict(result.intended_label_matches),
        "reason_codes": list(result.reason_codes),
    }


def _execution_row(execution: object) -> dict[str, object]:
    return {
        "status": getattr(execution, "status"),
        "message": getattr(execution, "message"),
        "value": jsonable(getattr(execution, "value")),
    }


def _write(handle: object, row: dict[str, object]) -> None:
    handle.write(json.dumps(jsonable(row), sort_keys=True) + "\n")
