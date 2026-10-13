

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


LANGUAGES = ("java", "typescript", "csharp", "python")
MISUSES = tuple(f"M{i:02d}" for i in range(1, 11))
STRUCTURES = ("callsite_helper", "callsite_inline", "variable_helper", "variable_inline")
RETRIEVAL_STRUCTURES = ("variable_inline", "callsite_inline", "variable_helper", "callsite_helper")
PAIRS = tuple(combinations(range(4), 2))
IDENTITY_FIELDS = ("original_id", "language", "misuse_id", "task_id", "sample_id")


def load_originals(root, limit):
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("--limit must be at least 1 original per cell")
    with (root / "d_index.jsonl").open(encoding="utf-8") as handle:
        originals = [json.loads(line) for line in handle]
    with (root / "delta_index.jsonl").open(encoding="utf-8") as handle:
        pairs = [json.loads(line) for line in handle]
    if not originals or len({r["original_id"] for r in originals}) != len(originals):
        raise ValueError("Expected one index record per original")
    for records in (originals, pairs):
        rows = [r["row"] for r in records]
        if any(type(row) is not int for row in rows) or sorted(rows) != list(range(len(records))):
            raise ValueError("Index rows must cover the corresponding array without gaps")
    pairs = {r["row"]: r for r in pairs}
    groups = {(language, misuse): [] for language in LANGUAGES for misuse in MISUSES}
    used = set()
    for original in originals:
        cell = original["language"], original["misuse_id"]
        if cell not in groups:
            raise ValueError(f"Unknown language/category in {original['original_id']}")
        prefix = f"{original['language']}__{original['misuse_id']}_{original['task_id']}_"
        if not original["original_id"].startswith(prefix):
            raise ValueError(f"Original ID does not match its metadata: {original['original_id']}")
        original["sample_id"] = original["original_id"][len(prefix):]
        rows = original["pair_rows"]
        if (not original["sample_id"] or len(rows) != 4 or len(set(rows)) != 4
                or any(type(row) is not int or row not in pairs for row in rows)
                or used.intersection(rows)):
            raise ValueError("Each original must have four distinct, unshared structure pairs")
        members = [pairs[row] for row in rows]
        if {r["structure"] for r in members} != set(STRUCTURES) or any(
                r[field] != original[field] for r in members for field in IDENTITY_FIELDS[:-1]):
            raise ValueError(f"Inconsistent structure-pair metadata for {original['original_id']}")
        by_structure = {r["structure"]: r["row"] for r in members}
        original["structure_rows"] = [by_structure[s] for s in STRUCTURES]
        used.update(rows)
        groups[cell].append(original)
    if len(used) != len(pairs):
        raise ValueError("Some structure pairs have no original in d_index.jsonl")
    selected = []
    for cell, members in groups.items():
        members.sort(key=lambda r: (r["task_id"], r["sample_id"]))
        if not members or (limit is not None and len(members) < limit):
            raise ValueError(f"{cell} has {len(members)} originals; all 40 cells are required")
        selected.extend(members[:limit])
    return selected, len(pairs)


def repair_metrics(delta):
    delta = np.asarray(delta, dtype=np.float32)
    if (delta.ndim != 3 or delta.shape[1] != 4 or not delta.shape[0]
            or not delta.shape[2] or not np.isfinite(delta).all()):
        raise ValueError("Expected finite repair vectors with shape originals x 4 structures x hidden width")
    # Raw float32 repairs; float64 scalar reductions match the histogram analysis.
    norms = np.sqrt(np.einsum("isd,isd->is", delta, delta, dtype=np.float64))
    if np.any(norms <= 0):
        raise ValueError("A zero repair vector has undefined cosine; originals cannot be silently omitted")
    cosines = np.column_stack([
        np.einsum("id,id->i", delta[:, a], delta[:, b], dtype=np.float64)
        / (norms[:, a] * norms[:, b]) for a, b in PAIRS
    ])
    cosines = np.clip(cosines, -1.0, 1.0)
    norm_cv = norms.std(axis=1, ddof=0) / norms.mean(axis=1)
    return {"norms": norms, "cosines": cosines, "norm_cv": norm_cv}


def unit_last(values):
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise ValueError("Category retrieval requires finite, nonzero mean vectors")
    return values / norms


def retrieve_categories(cell_means, show_progress=True):
    cell_means = np.asarray(cell_means)
    if cell_means.ndim != 4 or cell_means.shape[:3] != (4, 10, 4) or not cell_means.shape[3]:
        raise ValueError("Expected mean vectors for 4 languages x 10 categories x 4 structures")
    # Preserve the original retrieval's structure order for reference averaging.
    order = [STRUCTURES.index(s) for s in RETRIEVAL_STRUCTURES]
    values = unit_last(np.ascontiguousarray(cell_means[:, :, order, :], dtype=np.float64))
    rows = []
    with tqdm(total=160, desc="Cross-structure retrieval", unit="query", disable=not show_progress) as progress:
        for held_index, held_out in enumerate(RETRIEVAL_STRUCTURES):
            reference = unit_last(np.delete(values, held_index, axis=2).mean(axis=2))
            for language_index, language in enumerate(LANGUAGES):
                scores = values[language_index, :, held_index] @ reference[language_index].T
                for category_index, misuse in enumerate(MISUSES):
                    ranking = np.argsort(-scores[category_index], kind="stable")
                    rank = int(np.flatnonzero(ranking == category_index)[0]) + 1
                    alternatives = np.delete(scores[category_index], category_index)
                    rows.append({
                        "held_out_structure": held_out, "language": language, "misuse_id": misuse,
                        "predicted_misuse_id": MISUSES[int(ranking[0])], "rank": rank, "correct": rank == 1,
                        "correct_cosine": float(scores[category_index, category_index]),
                        "margin_over_best_incorrect": float(scores[category_index, category_index] - alternatives.max()),
                        **{f"cosine_{g}": float(scores[category_index, i]) for i, g in enumerate(MISUSES)},
                    })
                    progress.update(1)
    return rows


def analyze_structure(array, records, show_progress=True):
    if array.ndim != 2 or array.dtype != np.float32 or not array.shape[1]:
        raise ValueError("Expected a float32 matrix of structure-specific repair vectors")
    groups = {(language, misuse): [] for language in LANGUAGES for misuse in MISUSES}
    for record in records:
        groups[record["language"], record["misuse_id"]].append(record)
    means = np.empty((4, 10, 4, array.shape[1]), dtype=np.float32)
    agreement, summaries = [], []
    for (language, misuse), members in tqdm(groups.items(), desc="Structure agreement", unit="cell",
                                            disable=not show_progress):
        members.sort(key=lambda r: (r["task_id"], r["sample_id"]))
        if not members:
            raise ValueError(f"No originals for {language}/{misuse}")
        indices = np.asarray([r["structure_rows"] for r in members], dtype=int)
        if indices.shape != (len(members), 4) or np.any(indices < 0) or np.any(indices >= len(array)):
            raise ValueError("Each original must identify four valid repair rows")
        delta = np.ascontiguousarray(array[indices], dtype=np.float32)
        metrics = repair_metrics(delta)
        language_index, category_index = LANGUAGES.index(language), MISUSES.index(misuse)
        for s in range(4):
            # Mean raw repairs first. Original cell aggregation used a contiguous
            # float32 matrix for each structure, sorted by task and sample.
            means[language_index, category_index, s] = np.ascontiguousarray(delta[:, s, :]).mean(axis=0)
        for i, record in enumerate(members):
            agreement.append({
                **{field: record[field] for field in IDENTITY_FIELDS},
                **{f"cosine_{STRUCTURES[a]}__{STRUCTURES[b]}": float(metrics["cosines"][i, j])
                   for j, (a, b) in enumerate(PAIRS)},
                **{f"norm_{s}": float(metrics["norms"][i, j]) for j, s in enumerate(STRUCTURES)},
                "norm_cv": float(metrics["norm_cv"][i]),
            })
        summaries.append({"language": language, "misuse_id": misuse, "originals": len(members),
                          "mean_matched_cosine": float(metrics["cosines"].mean()),
                          "minimum_matched_cosine": float(metrics["cosines"].min()),
                          "mean_norm_cv": float(metrics["norm_cv"].mean())})
    retrieval = retrieve_categories(means, show_progress)
    for cell in summaries:
        queries = [r for r in retrieval if (r["language"], r["misuse_id"]) ==
                   (cell["language"], cell["misuse_id"])]
        cell.update({"retrieval_correct": sum(r["correct"] for r in queries), "retrieval_total": len(queries)})
    summary = {"originals": len(records), "structure_pairs": 6 * len(records), "cells": len(summaries),
               "structures": list(STRUCTURES),
               "cosine_pairs": [[STRUCTURES[a], STRUCTURES[b]] for a, b in PAIRS],
               "retrieval_correct": sum(r["correct"] for r in retrieval), "retrieval_total": len(retrieval),
               "mean_reciprocal_rank": float(np.mean([1.0 / r["rank"] for r in retrieval])),
               "mean_margin": float(np.mean([r["margin_over_best_incorrect"] for r in retrieval])),
               "by_cell": summaries}
    return agreement, retrieval, summary


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Model directory containing repair_vectors/ and layer_probes/")
    parser.add_argument("--output-dir", type=Path, help="Model directory for structure_agreement/; defaults to --input-dir")
    parser.add_argument("--layer", type=int, help="Zero-indexed layer; otherwise read step 03's selected layer")
    parser.add_argument("--limit", type=int, help="First N originals per cell, ordered by task and sample (40*N total)")
    args = parser.parse_args()
    layer = args.layer
    if layer is None:
        path = args.input_dir / "layer_probes/selected_layer.json"
        if not path.is_file():
            raise FileNotFoundError("Finish step 03 first, or provide --layer")
        layer = json.loads(path.read_text(encoding="utf-8"))["selected_layer"]
    if type(layer) is not int or layer < 0:
        raise ValueError("Layer must be a nonnegative integer")
    output = (args.output_dir if args.output_dir is not None else args.input_dir) / "structure_agreement"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Results already exist in {output}; choose another --output-dir")
    root = args.input_dir / "repair_vectors"
    records, pair_count = load_originals(root, args.limit)
    array = np.load(root / "delta" / f"layer_{layer:02d}.npy", mmap_mode="r", allow_pickle=False)
    if array.ndim != 2 or len(array) != pair_count:
        raise ValueError("The repair array must have one row per delta_index.jsonl record")
    print(f"Layer {layer}: {len(records)} originals, four structures each; CPU analysis.")
    agreement, retrieval, summary = analyze_structure(array, records)
    summary.update({"layer": layer,
                    "layer_source": "explicit" if args.layer is not None else "selected_layer.json",
                    "limit_per_cell": args.limit,
                    "full_source": args.limit is None and all(c["originals"] == 300 for c in summary["by_cell"])})
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "agreement.csv", agreement)
    write_csv(output / "retrieval.csv", retrieval)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Cross-structure category retrieval: {summary['retrieval_correct']}/{summary['retrieval_total']}")
    print(f"Saved agreement.csv, retrieval.csv, and summary.json to {output}")


if __name__ == "__main__":
    main()
