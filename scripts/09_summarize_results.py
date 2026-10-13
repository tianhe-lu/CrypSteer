import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from tqdm import tqdm


LANGUAGES = ("java", "typescript", "csharp", "python")
MISUSES = tuple(f"M{i:02d}" for i in range(1, 11))
COUNTS = ("n", "secure_count", "valid_count", "secure_valid_count",
          "unclassified_count", "provider_error_count")
METRICS = {"secure@1": "secure_count", "valid@1": "valid_count",
           "secure-valid@1": "secure_valid_count"}
GROUP_COLUMNS = {"overall": (), "by_language": ("language",),
                 "by_misuse": ("misuse_id",), "by_cell": ("language", "misuse_id")}
SETTING_COLUMNS = ("model", "setting", "method", "alpha", "source_scope", "layer", "sigma")
RESULT_COLUMNS = ("n", "expected_n", "complete", "cells", "expected_cells", *COUNTS[1:], *METRICS)


def find_settings(input_dir, names=None):
    if (input_dir / "evaluations.jsonl").is_file():
        directories = [input_dir]
    else:
        directories = sorted(path.parent for path in input_dir.glob("*/evaluations.jsonl"))
    if names:
        missing = set(names) - {path.name for path in directories}
        if missing:
            raise FileNotFoundError(f"No evaluations for settings: {sorted(missing)}")
        directories = [path for path in directories if path.name in names]
    if not directories:
        raise FileNotFoundError(f"No step-08 evaluations.jsonl found in {input_dir}")
    return directories


def summarize_setting(directory, limit=None, show_progress=True):
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1")
    saved = json.loads((directory / "settings.json").read_text(encoding="utf-8"))
    generation = saved["generation"]
    if saved["setting"] != directory.name:
        raise ValueError(f"Setting name differs from folder name: {directory}")
    metadata = {"model": generation["model_config"]["model_name"], "setting": saved["setting"],
                **{name: generation[name] for name in ("method", "alpha", "layer", "sigma")},
                "source_scope": generation.get("direction_selection", {}).get("source_scope", "full")}
    samples = generation["generation"]["samples_per_prompt"]
    seed_start = generation["generation"]["seed_start"]
    if type(samples) is not int or samples < 1 or type(seed_start) is not int:
        raise ValueError("Expected a positive samples_per_prompt and integer seed_start")
    prompts = {row["prompt_id"]: row for row in generation["prompts"]}
    if not prompts or len(prompts) != len(generation["prompts"]):
        raise ValueError("Expected nonempty prompts with unique prompt IDs")
    expected = Counter()
    for prompt in prompts.values():
        cell = prompt["language"], prompt["misuse_id"]
        if cell[0] not in LANGUAGES or cell[1] not in MISUSES:
            raise ValueError(f"Unknown language/category: {cell}")
        expected[cell] += samples
    counts = {cell: Counter() for cell in expected}
    seen = set()
    total = samples * len(prompts)
    with (directory / "evaluations.jsonl").open(encoding="utf-8") as handle, tqdm(
            total=min(total, limit) if limit is not None else total,
            desc=directory.name, unit="output", disable=not show_progress) as progress:
        for line in handle:
            if not line.endswith("\n"):
                break
            row = json.loads(line)
            identity = row["prompt_id"], row["sample_index"], row["seed"]
            if identity in seen:
                raise ValueError(f"Duplicate evaluation in {directory.name}: {identity}")
            if (row["prompt_id"] not in prompts or type(row["sample_index"]) is not int
                    or not 0 <= row["sample_index"] < samples
                    or row["seed"] != seed_start + row["sample_index"]):
                raise ValueError(f"Unexpected prompt/sample/seed: {identity}")
            if any(row[name] != metadata[name] for name in ("model", "setting", "method", "alpha", "layer", "sigma")):
                raise ValueError(f"Evaluation differs from its setting: {identity}")
            prompt = prompts[row["prompt_id"]]
            if any(row[name] != prompt[name] for name in ("language", "misuse_id", "task_id")):
                raise ValueError(f"Evaluation differs from its prompt: {identity}")
            label = row["target_detector_label"]
            if label not in ("secure", "misuse", "mixed", "unknown", None):
                raise ValueError(f"Unknown target security label: {label}")
            if type(row["program_valid"]) is not bool or not isinstance(row["provider_errors"], list):
                raise ValueError(f"Invalid validity or provider error fields: {identity}")
            secure, valid = label == "secure", row["program_valid"]
            cell_counts = counts[row["language"], row["misuse_id"]]
            cell_counts.update({"n": 1, "secure_count": int(secure), "valid_count": int(valid),
                                "secure_valid_count": int(secure and valid),
                                "unclassified_count": int(label is None),
                                "provider_error_count": int(bool(row["provider_errors"]))})
            seen.add(identity)
            progress.update(1)
            if limit is not None and len(seen) >= limit:
                break

    cells = [(language, misuse) for language in LANGUAGES for misuse in MISUSES
             if (language, misuse) in expected]

    def aggregate(selected, group):
        observed = [cell for cell in selected if counts[cell]["n"]]
        row = {**metadata, **group, **{name: sum(counts[cell][name] for cell in selected) for name in COUNTS},
               "expected_n": sum(expected[cell] for cell in selected),
               "cells": len(observed), "expected_cells": len(selected)}
        row["complete"] = row["n"] == row["expected_n"]
        for metric, count in METRICS.items():
            row[metric] = (100 * sum(counts[cell][count] / counts[cell]["n"] for cell in observed)
                           / len(observed)) if observed else None
        return row

    tables = {"overall": [aggregate(cells, {})], "by_language": [], "by_misuse": [], "by_cell": []}
    for language in LANGUAGES:
        selected = [cell for cell in cells if cell[0] == language]
        if selected:
            tables["by_language"].append(aggregate(selected, {"language": language}))
    for misuse in MISUSES:
        selected = [cell for cell in cells if cell[1] == misuse]
        if selected:
            tables["by_misuse"].append(aggregate(selected, {"misuse_id": misuse}))
    tables["by_cell"] = [aggregate([cell], {"language": cell[0], "misuse_id": cell[1]}) for cell in cells]
    return tables


def write_tables(tables, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, group_columns in GROUP_COLUMNS.items():
        with (output_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=(*SETTING_COLUMNS, *group_columns, *RESULT_COLUMNS))
            writer.writeheader()
            writer.writerows(tables[name])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Step-08 evaluation parent folder, or one setting folder")
    parser.add_argument("--output-dir", type=Path,
                        help="Summary folder; defaults to outputs/<model>/summary")
    parser.add_argument("--settings", nargs="+", help="Only these setting folder names")
    parser.add_argument("--limit", type=int, help="Read the first N complete evaluation records per setting")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    directories = find_settings(args.input_dir, args.settings)
    evaluation_root = args.input_dir.parent if (args.input_dir / "evaluations.jsonl").is_file() else args.input_dir
    output_dir = args.output_dir if args.output_dir is not None else evaluation_root.parent / "summary"
    tables = {name: [] for name in GROUP_COLUMNS}
    for directory in directories:
        setting_tables = summarize_setting(directory, args.limit)
        for name in tables:
            tables[name].extend(setting_tables[name])
        row = setting_tables["overall"][0]
        state = "complete" if row["complete"] else "partial"
        print(f"{row['setting']}: {row['n']}/{row['expected_n']} outputs, "
              f"{row['cells']}/{row['expected_cells']} cells ({state})")
    if len({row["model"] for row in tables["overall"]}) != 1:
        raise ValueError("Summarize one model at a time")
    write_tables(tables, output_dir)
    print(f"Saved overall.csv, by_language.csv, by_misuse.csv, by_cell.csv to {output_dir}")
    print("Rates are percentages with equal cell weights; complete=False marks partial coverage.")


if __name__ == "__main__":
    main()
