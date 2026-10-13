"""Export the paper's experiment tables from saved results to CSV, without refitting."""

import argparse
import csv
import json
import math
from pathlib import Path

from tqdm import tqdm


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
# Fixed checkpoint metadata; all measured values are read from experiment outputs.
MODELS = {
    "llama": ("Llama", "8B", 32),
    "codegemma": ("CodeGemma", "7B", 28),
    "qwen": ("Qwen", "7B", 28),
    "qwen_coder": ("Qwen-Coder", "7B", 28),
    "deepseek": ("DeepSeek", "16B", 27),
}
LANGUAGES = {"java": "Java", "typescript": "TypeScript", "csharp": "C#", "python": "Python"}
MISUSES = tuple(f"M{i:02d}" for i in range(1, 11))
METRICS = ("secure@1", "valid@1", "secure-valid@1")
METHODS = {"baseline": "Baseline", "cell": "Cell DiffMean", "pooled": "Pooled DiffMean",
           "normalized": "Normalized DiffMean", "crypsteer": "CrypSteer",
           "scs": "SCS-Code", "surgical": "Surgical Repair"}
DIRECTIONS = ("cell", "pooled", "normalized", "crypsteer")
TABLES = (
    "geometry_summary", "repair_heldout", "combination_prediction", "generation_main",
    "generation_source_exclusion", "generation_literature", "model_settings",
    "geometry_factors", "clustering", "key_contrasts", "language_method_comparison",
    "misuse_method_comparison", "literature_contrasts",
)


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value, places):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Table values must be finite")
    text = f"{value:.{places}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def require_complete(record, flag, context):
    if record.get(flag) not in (True, "True"):
        raise ValueError(f"{context}: complete paper results are required ({flag})")


def generation_index(root, filename, group_field=None):
    index = {}
    for row in read_csv(root / "summary" / filename):
        key = row["setting"], row[group_field] if group_field else ""
        if key in index:
            raise ValueError(f"{filename}: duplicate setting/group {key}")
        index[key] = row
    return index


def generation_row(index, model, setting, group="", n=6000, cells=40):
    row = index[setting, group]
    require_complete(row, "complete", f"{model}/{setting}/{group}")
    if row["model"] != model or int(row["n"]) != n or int(row["cells"]) != cells:
        raise ValueError(f"Unexpected model or coverage: {model}/{setting}/{group}")
    if setting == "baseline":
        expected = ("baseline", 0.0, "full")
    else:
        prefix, strength = setting.rsplit("_", 1)
        alpha = {"plus1sigma": 1.0, "minus1sigma": -1.0}[strength]
        if prefix.startswith("crypsteer_exclude_"):
            expected = ("crypsteer", alpha, prefix.removeprefix("crypsteer_"))
        else:
            expected = (prefix, alpha, "full")
    if (row["method"], float(row["alpha"]), row["source_scope"]) != expected:
        raise ValueError(f"Unexpected method, strength or fitting sources: {setting}")
    return row


def geometry_summary(root, model):
    saved = read_json(root / "geometry/summary.json")
    require_complete(saved, "full_source", "Geometry")
    fields = ["Model", "Same-category across-language mean cosine",
              "Different-category within-language mean cosine", "Category matches", "Norm ratio"]
    return fields, [[MODELS[model][0], number(saved["same_category_across_languages_cosine"], 3),
                     number(saved["different_categories_within_language_cosine"], 3),
                     f"{saved['retrieval_correct']}/{saved['retrieval_total']}",
                     number(saved["max_min_mean_original_norm_ratio"], 1)]]


def repair_heldout(root, model):
    saved = read_json(root / "heldout_alignment/summary.json")
    require_complete(saved, "full_population", "Held-out alignment")
    return ["Model", "Positive repairs (%)", "Mean cosine", "Positive cells"], [[
        MODELS[model][0], number(saved["positive_percent"], 2), number(saved["mean_cosine"], 3),
        f"{saved['positive_cells']}/{saved['cells']}"]]


def combination_prediction(root, model):
    saved = read_json(root / "combination_analysis/summary.json")
    require_complete(saved, "full_factorial", "Combinations")
    require_complete(saved, "direction_full_source", "Combination direction")
    records = read_csv(root / "combination_analysis/by_language.csv")
    index = {row["language"]: row for row in records}
    if len(records) != 4 or set(index) != set(LANGUAGES):
        raise ValueError("Combination prediction requires all four languages")
    row = [MODELS[model][0]]
    for language in LANGUAGES:
        record = index[language]
        if tuple(int(record[key]) for key in ("programs", "anchors", "prediction_programs")) != (1024, 11, 1013):
            raise ValueError("Expected 11 anchors and 1,013 prediction programs per language")
        row.append(number(record["prediction_r2"], 3))
    return ["Model", *LANGUAGES.values()], [row]


def generation_main(root, model):
    index = generation_index(root, "overall.csv")
    settings = [("baseline", "Baseline | alpha=0")]
    for sign, alpha in (("minus", "-1"), ("plus", "+1")):
        settings.extend((f"{method}_{sign}1sigma", f"{METHODS[method]} | alpha={alpha}")
                        for method in DIRECTIONS)
    records = [generation_row(index, model, setting) for setting, _ in settings]
    fields = ["Model", "Metric (%)", *(label for _, label in settings)]
    return fields, [[MODELS[model][0], metric, *(number(row[metric], 2) for row in records)]
                    for metric in METRICS]


def generation_source_exclusion(root, model):
    index = generation_index(root, "overall.csv")
    baseline = generation_row(index, model, "baseline")
    settings = ("crypsteer_plus1sigma", "crypsteer_exclude_language_plus1sigma",
                "crypsteer_exclude_misuse_plus1sigma", "crypsteer_exclude_both_plus1sigma")
    records = [generation_row(index, model, setting) for setting in settings]
    # Subtract unrounded percentages; do not subtract the rounded main-table cells.
    rows = [[MODELS[model][0], metric,
             *(number(float(row[metric]) - float(baseline[metric]), 2) for row in records)]
            for metric in ("secure@1", "secure-valid@1")]
    return ["Model", "Metric", "None gain (pp)", "Language gain (pp)",
            "Misuse gain (pp)", "Both gain (pp)"], rows


def generation_literature(root, model):
    index = generation_index(root, "overall.csv")
    rows = []
    for method in ("scs", "surgical", "crypsteer"):
        record = generation_row(index, model, f"{method}_plus1sigma")
        rows.append([MODELS[model][0], METHODS[method], *(number(record[m], 2) for m in METRICS)])
    return ["Model", "Method", *METRICS], rows


def model_settings(root, model):
    probe = read_json(root / "layer_probes/selected_layer.json")
    direction = read_json(root / "directions/selection.json")
    scale = read_json(root / "sigma.json")
    require_complete(probe, "full_corpus", "Layer probes")
    require_complete(direction, "full_source", "Direction")
    require_complete(scale, "full_source", "Scale")
    name, parameters, layers = MODELS[model]
    if (probe["selected_layer"] != direction["layer"] or direction["layer"] != scale["layer"]
            or probe["layers"] != list(range(layers))):
        raise ValueError(f"{model}: layer selection, direction, scale or checkpoint depth disagree")
    return ["Model", "Parameters", "Total layers", "Selected layer", "AUROC", "Selected PC", "sigma"], [[
        name, parameters, layers, probe["selected_layer"], number(probe["joint_auroc"], 3),
        direction["selected_pc"], number(scale["sigma"], 2)]]


def geometry_factors(root, model):
    saved = read_json(root / "variance/summary.json")
    require_complete(saved, "full_source", "Variance")
    return ["Model", "Category main effect (%)", "Language main effect (%)"], [[
        MODELS[model][0], number(saved["category_percent"], 2), number(saved["language_percent"], 2)]]


def clustering(root, model):
    saved = read_json(root / "clustering/summary.json")
    require_complete(saved, "full_source", "Clustering")
    if saved["cells"] != 40 or saved["config"]["cluster_count"] != 10:
        raise ValueError("Expected 40 cells clustered into ten groups")
    return ["Model", "Category ARI", "Language ARI"], [[MODELS[model][0],
        number(saved["category_ari"], 3), number(saved["language_ari"], 3)]]


def subgroup_table(root, model, group_field):
    is_language = group_field == "language"
    filename = "by_language.csv" if is_language else "by_misuse.csv"
    groups = LANGUAGES if is_language else {misuse: misuse for misuse in MISUSES}
    n, cells = (1500, 10) if is_language else (600, 4)
    index = generation_index(root, filename, group_field)
    methods = ("baseline", *DIRECTIONS)
    fields = ["Model", "Language" if is_language else "Misuse category"]
    fields += [f"{METHODS[method]} | {metric}" for method in methods for metric in METRICS]
    rows = []
    for group, label in groups.items():
        row = [MODELS[model][0], label]
        for method in methods:
            setting = "baseline" if method == "baseline" else f"{method}_plus1sigma"
            saved = generation_row(index, model, setting, group, n, cells)
            row.extend(number(saved[metric], 2) for metric in METRICS)
        rows.append(row)
    return fields, rows


def contrast_table(root, model, family):
    comparators = ("baseline", "cell", "pooled", "normalized") if family == "main" else ("scs", "surgical")
    family_size = 10 if family == "main" else 6
    records = read_csv(root / "paired_statistics/contrasts.csv")
    index = {}
    for row in records:
        if row["family"] != family:
            continue
        key = row["method_a"], row["method_b"], row["metric"]
        if key in index:
            raise ValueError(f"Duplicate saved contrast: {key}")
        index[key] = row
    fields = ["Model", "Comparator", "Holm family", "Family comparisons"]
    fields += [f"{metric} | {part}" for metric in METRICS
               for part in ("difference (pp)", "CI low (pp)", "CI high (pp)", "Holm p", "significant")]
    rows = []
    for comparator in comparators:
        setting = "baseline" if comparator == "baseline" else f"{comparator}_plus1sigma"
        result = [MODELS[model][0], METHODS[comparator], family, family_size]
        for metric in METRICS:
            key = "crypsteer_plus1sigma", setting, metric
            reverse = False
            if key not in index:
                key = setting, "crypsteer_plus1sigma", metric
                reverse = True
            row = index[key]
            require_complete(row, "complete", f"{model}/{family}/{metric}")
            if (row["model"] != model or int(row["family_size"]) != family_size
                    or tuple(int(row[k]) for k in ("n", "cells", "prompts_per_cell", "samples_per_prompt"))
                    != (6000, 40, 5, 30)):
                raise ValueError("Contrast has unexpected model, Holm family or paired coverage")
            delta, low, high = (float(row[k]) for k in ("difference_pp", "ci_low_pp", "ci_high_pp"))
            if reverse:
                delta, low, high = -delta, -high, -low
            if low > high or not 0 <= float(row["p_holm"]) <= 1:
                raise ValueError("Invalid saved confidence interval or Holm p-value")
            if row["significant"] not in ("True", "False"):
                raise ValueError("Expected a saved boolean significance flag")
            # Preserve the original 10/6-comparison correction, including undisplayed contrasts.
            result.extend([number(delta, 2), number(low, 2), number(high, 2),
                           row["p_holm"], row["significant"]])
        rows.append(result)
    return fields, rows


def build_table(table, root, model):
    if table == "language_method_comparison":
        return subgroup_table(root, model, "language")
    if table == "misuse_method_comparison":
        return subgroup_table(root, model, "misuse_id")
    if table in ("key_contrasts", "literature_contrasts"):
        return contrast_table(root, model, "main" if table == "key_contrasts" else "literature")
    builders = {"geometry_summary": geometry_summary, "repair_heldout": repair_heldout,
                "combination_prediction": combination_prediction, "generation_main": generation_main,
                "generation_source_exclusion": generation_source_exclusion,
                "generation_literature": generation_literature, "model_settings": model_settings,
                "geometry_factors": geometry_factors, "clustering": clustering}
    return builders[table](root, model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", "--input-dir", nargs="+", type=Path, required=True,
                        help="Model result folders, in display order")
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS),
                        help="Model IDs if input folder names differ, one per folder")
    parser.add_argument("--tables", nargs="+", choices=TABLES, default=list(TABLES),
                        help="Tables to export (default: all thirteen)")
    parser.add_argument("--output-dir", type=Path,
                        help="CSV folder; default: model/tables for one model or outputs/tables for several")
    parser.add_argument("--limit", type=int, help="Use the first N input models, keeping their complete table rows")
    args = parser.parse_args()
    models = args.models or [root.name for root in args.input_dirs]
    if len(models) != len(args.input_dirs) or any(model not in MODELS for model in models):
        parser.error("Use known model folder names, or provide one --models ID per input folder")
    if len(set(models)) != len(models):
        parser.error("Provide each model once")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    roots, models = args.input_dirs[:args.limit], models[:args.limit]
    tables = list(dict.fromkeys(args.tables))
    output = args.output_dir or ((roots[0] if len(roots) == 1 else ARTIFACT_ROOT / "outputs") / "tables")
    exports = []
    # Read every requested table before writing, so a missing input does not leave a partial export.
    with tqdm(total=len(tables) * len(models), desc="Read table results", unit="model-table") as progress:
        for table in tables:
            rows, header = [], None
            for root, model in zip(roots, models):
                try:
                    fields, values = build_table(table, root, model)
                except (OSError, KeyError, ValueError) as exc:
                    parser.error(f"{model}/{table}: {exc}. Supply the required stage outputs or select --tables.")
                if header is not None and header != fields:
                    raise ValueError(f"Inconsistent columns for {table}")
                header = fields
                rows.extend(values)
                progress.update(1)
            exports.append((table, header, rows))
    output.mkdir(parents=True, exist_ok=True)
    for table, fields, rows in tqdm(exports, desc="Write CSV tables", unit="table"):
        path = output / f"table_{table}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            writer.writerows(rows)
    print(f"Saved {len(exports)} tables for {', '.join(MODELS[m][0] for m in models)}: {output}")
    print("Rates are percentages; differences and CI bounds are percentage points. Holm values are unchanged.")


if __name__ == "__main__":
    main()
