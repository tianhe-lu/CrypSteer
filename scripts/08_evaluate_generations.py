import argparse
import json
from pathlib import Path
import sys

from tqdm import tqdm


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_ROOT = ARTIFACT_ROOT / "evaluator"
sys.path.insert(0, str(EVALUATOR_ROOT))

from crypto_steering.config.models import (
    ValidationConfig, ValidationDatasetAuditConfig, ValidationExecutionConfig,
    ValidationFrontendConfig, ValidationPolicyConfig, ValidationProbeAuditConfig,
    ValidationProviderConfig,
)
from crypto_steering.validation.models import ProgramInput, jsonable
from crypto_steering.validation.reporting import _program_row
from crypto_steering.validation.runner import iter_validation_results


SAMPLE_FIELDS = ("model", "prompt_id", "language", "misuse_id", "task_id", "method",
                 "alpha", "layer", "sigma", "effective_coefficient", "sample_index", "seed")
RESULT_FIELDS = ("frontend_valid", "import_valid", "target_task_valid",
                 "target_detector_label", "secondary_misuse_ids", "program_valid",
                 "validity_rules", "reason_codes")
EXTENSIONS = {"java": ".java", "typescript": ".ts", "csharp": ".cs", "python": ".py"}


def load_config(path, workers=None):
    settings = json.loads(path.read_text(encoding="utf-8"))
    if workers is not None:
        settings["execution"]["workers"] = workers
    execution = settings["execution"]
    if execution["workers"] < 1 or execution["analysis_timeout_seconds"] < 1:
        raise ValueError("Workers and analysis timeout must be positive")
    if settings["policy"]["label_minimum_iterations"] != 1000:
        raise ValueError("The detector uses the paper's 1,000-iteration policy")
    frontends = []
    for language, values in settings["frontends"].items():
        values = dict(values)
        for name in ("compiler_script", "tsconfig_path", "tooling_project"):
            if name in values:
                values[name] = str((EVALUATOR_ROOT / values[name]).resolve())
        frontends.append(ValidationFrontendConfig(language, tuple(values.items())))
    config = ValidationConfig(
        schema_version=1, config_id="generation_evaluation", selected_provider="internal",
        providers=(ValidationProviderConfig("internal", ("frontend", "task", "misuse")),),
        policy=ValidationPolicyConfig(
            **settings["policy"], rewrite_rules=(), structures=(), task_families=()),
        frontends=tuple(frontends), execution=ValidationExecutionConfig(**execution),
        dataset_audit=ValidationDatasetAuditConfig(0, 0, 0, (), ()),
        probe_audit=ValidationProbeAuditConfig(()),
    )
    return config, settings


def find_settings(input_dir, names=None):
    if (input_dir / "generations.jsonl").is_file():
        directories = [input_dir]
    else:
        directories = sorted(path.parent for path in input_dir.glob("*/generations.jsonl"))
    if names:
        missing = set(names) - {path.name for path in directories}
        if missing:
            raise FileNotFoundError(f"No generated outputs for: {sorted(missing)}")
        directories = [path for path in directories if path.name in names]
    if not directories:
        raise FileNotFoundError(f"No step-07 generations.jsonl found in {input_dir}")
    return directories


def sample_key(row):
    return row["prompt_id"], row["sample_index"], row["seed"]


def load_generations(directory, limit):
    settings = json.loads((directory / "settings.json").read_text(encoding="utf-8"))
    records, seen = [], set()
    with (directory / "generations.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.endswith("\n"):
                break
            row = json.loads(line)
            key = sample_key(row)
            if key in seen:
                raise ValueError(f"Duplicate generated output in {directory}: {key}")
            if (row["model"] != settings["model_config"]["model_name"]
                    or row["method"] != settings["method"] or row["alpha"] != settings["alpha"]):
                raise ValueError(f"Generated output differs from its setting: {key}")
            if row["language"] not in EXTENSIONS or not isinstance(row["code"], str):
                raise ValueError(f"Unsupported language or missing source: {key}")
            if Path(row["prompt_id"]).name != row["prompt_id"]:
                raise ValueError(f"Expected a prompt ID, not a path: {key}")
            seen.add(key)
            records.append(row)
            if limit is not None and len(records) >= limit:
                break
    return records, settings


def source_relative_path(row):
    return Path(row["language"]) / row["prompt_id"] / str(row["sample_index"]).zfill(3) / (
        "source" + EXTENSIONS[row["language"]])


def read_existing(output_dir, settings, resume):
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"Results already exist in {output_dir}; use --resume or another --output-dir")
    settings_path = output_dir / "settings.json"
    results_path = output_dir / "evaluations.jsonl"
    if settings_path.exists():
        previous = json.loads(settings_path.read_text(encoding="utf-8"))
        if previous != settings:
            raise ValueError(f"Evaluation or generation settings differ in {output_dir}")
    elif results_path.exists():
        raise FileNotFoundError(f"Cannot resume {results_path} without settings.json")
    existing = {}
    if results_path.exists():
        with results_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = sample_key(row)
                if key in existing:
                    raise ValueError(f"Duplicate evaluation in {results_path}: {key}")
                existing[key] = row
    return existing


def pending_records(records, existing, output_dir):
    pending = []
    for row in records:
        previous = existing.get(sample_key(row))
        if previous is None:
            pending.append(row)
            continue
        if any(previous[name] != row[name] for name in SAMPLE_FIELDS):
            raise ValueError(f"Sample settings changed for {sample_key(row)}")
        source = output_dir / "work/sources" / source_relative_path(row)
        if not source.exists() or source.read_bytes().decode("utf-8") != row["code"]:
            raise ValueError(f"Saved evaluation source differs for {sample_key(row)}")
    return pending


def make_program(row, setting, output_dir):
    relative = source_relative_path(row)
    path = output_dir / "work/sources" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(row["code"], encoding="utf-8")
    program_id = f"{row['model']}/{setting}/{row['prompt_id']}/{row['sample_index']:03d}"
    return ProgramInput(
        program_id=program_id, relative_path=relative.as_posix(), source_path=path,
        language=row["language"], source=path.read_text(encoding="utf-8"), corpus_kind="generated",
        validation_mode="strict", analysis_profile="whole_program",
        target_misuse_id=row["misuse_id"], task_id=row["task_id"],
        metadata={name: row[name] for name in SAMPLE_FIELDS},
    )


def evaluation_row(result, row, setting):
    reported = _program_row(result)
    return {
        **{name: row[name] for name in SAMPLE_FIELDS}, "setting": setting,
        "program_id": result.program.program_id,
        **{name: reported[name] for name in RESULT_FIELDS},
        "provider_errors": list(result.provider_errors),
        "frontend": jsonable(result.frontend), "imports": jsonable(result.imports),
        "tasks": jsonable(result.tasks), "detectors": jsonable(result.detectors),
    }


def empty_evaluation(row, setting, output_dir):
    path = output_dir / "work/sources" / source_relative_path(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(row["code"], encoding="utf-8")
    skipped = {"status": "skipped", "value": None, "message": "empty_generated_source"}
    return {
        **{name: row[name] for name in SAMPLE_FIELDS}, "setting": setting,
        "program_id": f"{row['model']}/{setting}/{row['prompt_id']}/{row['sample_index']:03d}",
        "frontend_valid": False, "import_valid": False, "target_task_valid": False,
        "target_detector_label": None, "secondary_misuse_ids": [], "program_valid": False,
        "validity_rules": {}, "reason_codes": ["empty_generated_source"], "provider_errors": [],
        "frontend": skipped, "imports": skipped, "tasks": {row["misuse_id"]: skipped},
        "detectors": {f"M{i:02d}": skipped for i in range(1, 11)},
    }


def evaluate_setting(directory, output_dir, config, config_settings, limit, resume, batch_size):
    records, generation = load_generations(directory, limit)
    if not records:
        print(f"{directory.name}: no complete generated records yet")
        return
    settings = {
        "setting": directory.name, "evaluation": config_settings,
        "generation": {key: value for key, value in generation.items() if key != "runtime"},
    }
    existing = read_existing(output_dir, settings, resume)
    pending = pending_records(records, existing, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    errors = 0
    with (output_dir / "evaluations.jsonl").open("a", encoding="utf-8") as handle, tqdm(
            total=len(records), initial=len(records) - len(pending), desc=directory.name,
            unit="program") as progress:
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            for row in batch:
                if not row["code"].strip():
                    result = empty_evaluation(row, directory.name, output_dir)
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                    progress.update()
            batch = [row for row in batch if row["code"].strip()]
            if not batch:
                continue
            programs = [make_program(row, directory.name, output_dir) for row in batch]
            inputs = {program.program_id: row for program, row in zip(programs, batch)}
            work_dir = output_dir / "work/checks" / f"batch_{len(existing) + start:06d}"
            for result in iter_validation_results(programs, config, work_dir=work_dir):
                row = evaluation_row(result, inputs[result.program.program_id], directory.name)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                errors += bool(row["provider_errors"])
                progress.update()
    print(f"{directory.name}: {len(pending)} evaluated, {len(records) - len(pending)} reused; "
          f"{errors} new outputs with provider errors. Results: {output_dir / 'evaluations.jsonl'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Step-07 generation parent folder, or one setting folder")
    parser.add_argument("--output-dir", type=Path,
                        help="Evaluation parent folder; each setting gets a subfolder")
    parser.add_argument("--config", type=Path, default=ARTIFACT_ROOT / "scripts/config/evaluation.json")
    parser.add_argument("--settings", nargs="+", help="Only these setting folder names")
    parser.add_argument("--workers", type=int, help="Override CPU workers in evaluation.json")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="Programs per frontend/analysis batch")
    parser.add_argument("--limit", type=int, help="First N completed outputs per setting")
    parser.add_argument("--resume", action="store_true", help="Keep saved evaluations; process new outputs")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--batch-size and --limit must be positive")
    input_dir = args.input_dir.resolve()
    directories = find_settings(input_dir, args.settings)
    generation_parent = input_dir.parent if (input_dir / "generations.jsonl").is_file() else input_dir
    output_parent = (args.output_dir or generation_parent.parent / "evaluation").resolve()
    config, settings = load_config(args.config, args.workers)
    for directory in directories:
        evaluate_setting(directory, output_parent / directory.name, config, settings,
                         args.limit, args.resume, args.batch_size)


if __name__ == "__main__":
    main()
