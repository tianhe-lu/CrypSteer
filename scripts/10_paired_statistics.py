import argparse
import csv
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


LANGUAGES = ("java", "typescript", "csharp", "python")
MISUSES = tuple(f"M{i:02d}" for i in range(1, 11))
METRICS = ("secure@1", "valid@1", "secure-valid@1")
FAMILIES = {
    "main": ("baseline", "cell_plus1sigma", "pooled_plus1sigma",
             "normalized_plus1sigma", "crypsteer_plus1sigma"),
    "literature": ("baseline", "scs_plus1sigma", "surgical_plus1sigma", "crypsteer_plus1sigma"),
}
COLUMNS = ("model", "family", "family_size", "method_a", "method_b", "metric", "n",
           "cells", "prompts_per_cell", "samples_per_prompt", "complete", "rate_a", "rate_b",
           "difference_pp", "ci_low_pp", "ci_high_pp", "p_value", "p_holm", "significant")


def load_panel(input_dir, settings, limit=None, show_progress=True):
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("--limit must be a positive number of samples per prompt")
    cells = [(language, misuse) for language in LANGUAGES for misuse in MISUSES]
    values, info, common_prompts, common_sampling = None, None, None, None
    common_model, common_evaluation = None, None
    steering_settings = None
    for method_index, setting in enumerate(settings):
        directory = input_dir / setting
        saved = json.loads((directory / "settings.json").read_text(encoding="utf-8"))
        generation = saved["generation"]
        model = generation["model_config"]["model_name"]
        evaluation = {field: saved["evaluation"][field] for field in ("frontends", "policy")}
        method = "baseline" if setting == "baseline" else setting.removesuffix("_plus1sigma")
        alpha = 0.0 if method == "baseline" else 1.0
        if (saved["setting"] != setting or generation["method"] != method
                or generation["alpha"] != alpha
                or generation.get("direction_selection", {}).get("source_scope", "full") != "full"):
            raise ValueError(f"Expected full-source +1 sigma setting (or unsteered baseline): {setting}")
        if method != "baseline":
            current = (generation["layer"], generation["sigma"])
            if steering_settings is not None and current != steering_settings:
                raise ValueError("Methods must use the same layer and sigma")
            steering_settings = current
        sampling = generation["generation"]
        total_samples, seed_start = sampling["samples_per_prompt"], sampling["seed_start"]
        if type(total_samples) is not int or total_samples < 1 or type(seed_start) is not int:
            raise ValueError("Expected a positive samples_per_prompt and integer seed_start")
        samples = min(total_samples, limit) if limit is not None else total_samples
        prompts = {row["prompt_id"]: {field: row.get(field) for field in
                   ("language", "misuse_id", "task_id", "generation_prompt")}
                   for row in generation["prompts"]}
        if len(prompts) != 200 or len(prompts) != len(generation["prompts"]):
            raise ValueError(f"Expected 200 distinct prompts: {setting}")
        if set((row["language"], row["misuse_id"]) for row in prompts.values()) != set(cells):
            raise ValueError(f"Expected all 40 language/category cells: {setting}")
        cell_prompts = [sorted(prompt for prompt, row in prompts.items()
                              if (row["language"], row["misuse_id"]) == cell) for cell in cells]
        if any(len(items) != 5 for items in cell_prompts):
            raise ValueError(f"Expected five prompts in every cell: {setting}")
        if values is None:
            common_prompts, common_sampling = prompts, sampling
            common_model, common_evaluation = generation["model_config"], evaluation
            values = np.empty((len(settings), len(METRICS), 40, 5, samples), dtype=np.int8)
            info = {"model": model, "n": 200 * samples, "expected_n": 200 * total_samples,
                    "complete": samples == total_samples, "cells": 40, "prompts_per_cell": 5,
                    "samples_per_prompt": samples, "seed_start": seed_start}
        elif (generation["model_config"] != common_model or evaluation != common_evaluation
              or prompts != common_prompts or sampling != common_sampling):
            raise ValueError(f"Model, prompts, generation or evaluation policy settings differ: {setting}")
        positions = {prompt: (c, p) for c, items in enumerate(cell_prompts) for p, prompt in enumerate(items)}
        seen = set()
        with (directory / "evaluations.jsonl").open(encoding="utf-8") as handle, tqdm(
                total=info["n"], desc=setting, unit="output", disable=not show_progress) as progress:
            for line in handle:
                if not line.endswith("\n"):
                    break
                row = json.loads(line)
                prompt, sample, seed = row["prompt_id"], row["sample_index"], row["seed"]
                if (prompt not in prompts or type(sample) is not int or not 0 <= sample < total_samples
                        or seed != seed_start + sample):
                    raise ValueError(f"Unexpected prompt/sample/seed in {setting}: {(prompt, sample, seed)}")
                if sample >= samples:
                    continue
                identity = (prompt, sample, seed)
                if identity in seen:
                    raise ValueError(f"Duplicate paired output in {setting}: {identity}")
                expected = {"model": model, "setting": setting, "method": method, "alpha": alpha,
                            "layer": generation["layer"], "sigma": generation["sigma"]}
                if any(row[field] != value for field, value in expected.items()):
                    raise ValueError(f"Output differs from its setting: {setting}/{identity}")
                if any(row[field] != prompts[prompt][field] for field in ("language", "misuse_id", "task_id")):
                    raise ValueError(f"Output differs from its prompt: {setting}/{identity}")
                label, valid = row["target_detector_label"], row["program_valid"]
                if label not in ("secure", "misuse", "mixed", "unknown", None) or type(valid) is not bool:
                    raise ValueError(f"Invalid security/validity label: {setting}/{identity}")
                secure = label == "secure"
                c, p = positions[prompt]
                values[method_index, :, c, p, sample] = (secure, valid, secure and valid)
                seen.add(identity)
                progress.update(1)
        if len(seen) != info["n"]:
            raise ValueError(f"Incomplete paired grid in {setting}: {len(seen)}/{info['n']}. "
                             "Finish evaluation or use --limit for a complete prefix of seeds.")
    if values is None or steering_settings is None:
        raise ValueError("Expected a complete comparison family")
    info.update({"layer": steering_settings[0], "sigma": steering_settings[1]})
    return values, cells, info


def bootstrap_rates(values, replicates, random_seed, batch_size=250, show_progress=True):
    data = np.asarray(values, dtype=np.int8)
    methods, metrics, cells, prompts, samples = data.shape
    rng = np.random.default_rng(random_seed)
    rates = np.empty((methods, metrics, cells, replicates), dtype=np.float64)
    with tqdm(total=cells * replicates, desc="Paired bootstrap", unit="draw",
              disable=not show_progress) as progress:
        for cell in range(cells):
            cell_values = data[:, :, cell]
            for start in range(0, replicates, batch_size):
                stop = min(replicates, start + batch_size)
                size = stop - start
                prompt_indexes = rng.integers(0, prompts, size=(size, prompts))
                sample_indexes = rng.integers(0, samples, size=(size, prompts, samples))
                selected = cell_values[:, :, prompt_indexes, :]
                selected = np.take_along_axis(selected, sample_indexes[None, None], axis=-1)
                rates[:, :, cell, start:stop] = selected.mean(axis=(-2, -1), dtype=np.float64)
                progress.update(size)
    return rates.mean(axis=2, dtype=np.float64)


def sign_flip_p_values(prompt_effects, replicates, random_seed, batch_size=10000, show_progress=True):
    effects = np.asarray(prompt_effects, dtype=np.float64)
    flattened = effects.reshape(effects.shape[0], -1)
    observed = flattened.mean(axis=1)
    exceedances = np.zeros(effects.shape[0], dtype=np.int64)
    rng = np.random.default_rng(random_seed)
    with tqdm(total=replicates, desc="Prompt sign flips", unit="draw", disable=not show_progress) as progress:
        for start in range(0, replicates, batch_size):
            size = min(batch_size, replicates - start)
            signs = rng.integers(0, 2, size=(size, flattened.shape[1]), dtype=np.int8) * 2 - 1
            null_statistics = signs @ flattened.T / flattened.shape[1]
            exceedances += np.count_nonzero(
                np.abs(null_statistics) >= np.abs(observed)[None, :] - 1e-15, axis=0)
            progress.update(size)
    return (exceedances + 1.0) / (replicates + 1.0)


def holm_adjust(p_values):
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = np.ones(len(p_values), dtype=np.float64)
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, min(1.0, (len(order) - rank) * float(p_values[index])))
        adjusted[index] = running_max
    return adjusted


def analyze_family(values, settings, family, model, config, info, show_progress=True):
    if tuple(settings) != FAMILIES[family]:
        raise ValueError(f"Use the full {family} family in its specified method order")
    seeds = config["seeds"][model][family]
    bootstrap = bootstrap_rates(values, config["bootstrap_replicates"], seeds["bootstrap"],
                                config["bootstrap_batch_size"], show_progress)
    tests = [(later, earlier, metric) for earlier, later in combinations(range(len(settings)), 2)
             for metric in range(len(METRICS))]
    effects = np.stack([(values[a, o] - values[b, o]).mean(axis=-1, dtype=np.float64)
                        for a, b, o in tests])
    p_values = sign_flip_p_values(effects, config["sign_flip_replicates"], seeds["sign_flip"],
                                 config["sign_flip_batch_size"], show_progress)
    adjusted = np.empty(len(tests))
    for metric in range(len(METRICS)):
        indexes = [i for i, (_, _, outcome) in enumerate(tests) if outcome == metric]
        adjusted[indexes] = holm_adjust(p_values[indexes])
    tail = (1 - config["confidence_level"]) / 2
    rows = []
    for index, (a, b, metric) in enumerate(tests):
        low, high = np.quantile(bootstrap[a, metric] - bootstrap[b, metric], (tail, 1 - tail))
        rate_a, rate_b = values[a, metric].mean(), values[b, metric].mean()
        rows.append({"model": model, "family": family, "family_size": len(tests) // len(METRICS),
                     "method_a": settings[a], "method_b": settings[b], "metric": METRICS[metric],
                     **{name: info[name] for name in
                        ("n", "cells", "prompts_per_cell", "samples_per_prompt", "complete")},
                     "rate_a": 100 * float(rate_a), "rate_b": 100 * float(rate_b),
                     "difference_pp": 100 * float(rate_a - rate_b),
                     "ci_low_pp": 100 * float(low), "ci_high_pp": 100 * float(high),
                     "p_value": float(p_values[index]), "p_holm": float(adjusted[index]),
                     "significant": bool(adjusted[index] < 0.05)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Step-08 evaluation parent for one model")
    parser.add_argument("--output-dir", type=Path, help="Defaults to outputs/<model>/paired_statistics")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config/statistics.json")
    parser.add_argument("--families", choices=tuple(FAMILIES), nargs="+", default=list(FAMILIES))
    parser.add_argument("--limit", type=int, help="Use the first N seeds per prompt, retaining all 200 prompts")
    parser.add_argument("--bootstrap-replicates", type=int, help="Override for a short test")
    parser.add_argument("--sign-flip-replicates", type=int, help="Override for a short test")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for field in ("bootstrap_replicates", "sign_flip_replicates"):
        override = getattr(args, field)
        if override is not None:
            config[field] = override
    for field in ("bootstrap_replicates", "sign_flip_replicates", "bootstrap_batch_size", "sign_flip_batch_size"):
        if type(config[field]) is not int or config[field] < 1:
            parser.error(f"{field} must be a positive integer")
    if not 0 < config["confidence_level"] < 1:
        parser.error("confidence_level must be between zero and one")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    families = list(dict.fromkeys(args.families))
    settings = tuple(dict.fromkeys(setting for family in families for setting in FAMILIES[family]))
    values, cells, info = load_panel(args.input_dir, settings, args.limit)
    model = info["model"]
    if model not in config["seeds"]:
        parser.error(f"Add explicit resampling seeds for {model} to the statistics config")
    rows, cell_orders = [], {}
    for family in families:
        methods = FAMILIES[family]
        ordered_cells = sorted(cells) if family == "literature" else cells
        cell_orders[family] = ordered_cells
        selected = values[[settings.index(method) for method in methods]]
        selected = selected[:, :, [cells.index(cell) for cell in ordered_cells]]
        print(f"{model}: {family}, {info['n']} paired outputs per setting")
        rows.extend(analyze_family(selected, methods, family, model, config, info))
    output_dir = args.output_dir if args.output_dir is not None else args.input_dir.parent / "paired_statistics"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "contrasts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    saved = {"model": model, "input_dir": str(args.input_dir.resolve()), "families": families,
             "limit": args.limit, "config": config, "panel": info, "cell_order": cell_orders,
             "contrast_direction": "method_a minus method_b", "rate_unit": "percent",
             "difference_unit": "percentage points"}
    (output_dir / "settings.json").write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} contrasts to {output_dir}")
    if not info["complete"]:
        print("Limited seed subset: these statistics are for testing, not the full paper results.")


if __name__ == "__main__":
    main()
