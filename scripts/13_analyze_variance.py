"""Measure category and language main effects among unit cell repair directions."""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


LANGUAGES = ("java", "typescript", "csharp", "python")
MISUSES = tuple(f"M{i:02d}" for i in range(1, 11))
EPSILON = 1e-12


def load_originals(root, limit):
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("--limit must be at least 1 original per cell")
    with (root / "d_index.jsonl").open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    if not records or len({r["original_id"] for r in records}) != len(records):
        raise ValueError("Expected one index record per original")
    rows = [r["row"] for r in records]
    if any(type(row) is not int for row in rows) or sorted(rows) != list(range(len(records))):
        raise ValueError("Index rows must cover the repair array without gaps")
    groups = {(language, misuse): [] for language in LANGUAGES for misuse in MISUSES}
    for record in records:
        cell = record["language"], record["misuse_id"]
        if cell not in groups:
            raise ValueError(f"Unknown language/category in {record['original_id']}")
        prefix = f"{record['language']}__{record['misuse_id']}_{record['task_id']}_"
        if not record["original_id"].startswith(prefix):
            raise ValueError(f"Original ID does not match its metadata: {record['original_id']}")
        record["sample_id"] = record["original_id"][len(prefix):]
        if not record["sample_id"]:
            raise ValueError(f"Missing sample identity: {record['original_id']}")
        groups[cell].append(record)
    selected = []
    counts = []
    for cell, members in groups.items():
        members.sort(key=lambda r: (r["task_id"], r["sample_id"]))
        if not members or (limit is not None and len(members) < limit):
            raise ValueError(f"{cell} has {len(members)} originals; all 40 cells are required")
        selected.extend(members[:limit])
        counts.append(len(members[:limit]))
    if len(set(counts)) != 1:
        raise ValueError("Use the same number of originals in every language/category cell")
    return selected, len(records)


def construct_cell_directions(array, records, show_progress=True):
    if array.ndim != 2 or array.dtype != np.float32 or not array.shape[1]:
        raise ValueError("Expected a float32 matrix of original-level repair vectors d")
    groups = {(language, misuse): [] for language in LANGUAGES for misuse in MISUSES}
    for record in records:
        groups[record["language"], record["misuse_id"]].append(record)
    means = []
    for (language, misuse), members in tqdm(groups.items(), desc="Cell repair means", unit="cell",
                                            disable=not show_progress):
        members.sort(key=lambda r: (r["task_id"], r["sample_id"]))
        if not members:
            raise ValueError(f"No originals for {language}/{misuse}")
        indices = np.asarray([r["row"] for r in members], dtype=int)
        if np.any(indices < 0) or np.any(indices >= len(array)):
            raise ValueError("Original index refers to a missing repair row")
        d = np.ascontiguousarray(array[indices], dtype=np.float32)
        if not np.isfinite(d).all():
            raise ValueError(f"Nonfinite repair vector in {language}/{misuse}")
        # Match the original aggregation: average raw d in float32, then
        # normalize each cell mean in float64, not each individual repair.
        means.append(d.mean(axis=0))
    values = np.asarray(means, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= EPSILON):
        raise ValueError("Cell directions require finite, nonzero mean repair vectors")
    return (values / norms).reshape(len(LANGUAGES), len(MISUSES), array.shape[1])


def decompose_variance(grid):
    values = np.asarray(grid, dtype=np.float64)
    if (values.ndim != 3 or values.shape[:2] != (len(LANGUAGES), len(MISUSES))
            or not values.shape[2] or not np.isfinite(values).all()):
        raise ValueError("Expected a finite grid of 4 languages x 10 categories x hidden width")
    grand = values.mean(axis=(0, 1))
    language_means = values.mean(axis=1)
    category_means = values.mean(axis=0)
    # T, SS_L, and SS_G in the paper. Every cell has equal weight.
    total = float(np.square(values - grand).sum())
    language = float(len(MISUSES) * np.square(language_means - grand).sum())
    category = float(len(LANGUAGES) * np.square(category_means - grand).sum())
    if total <= EPSILON:
        raise ValueError("Main-effect shares are undefined when total centered variation is zero")
    return {
        "total_sum_squares": total,
        "category_sum_squares": category,
        "language_sum_squares": language,
        "category_fraction": category / total,
        "language_fraction": language / total,
        "category_percent": 100 * (category / total),
        "language_percent": 100 * (language / total),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Model directory containing repair_vectors/ and layer_probes/")
    parser.add_argument("--output-dir", type=Path,
                        help="Model directory for variance/; defaults to --input-dir")
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
    output = (args.output_dir if args.output_dir is not None else args.input_dir) / "variance"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Results already exist in {output}; choose another --output-dir")
    root = args.input_dir / "repair_vectors"
    records, original_count = load_originals(root, args.limit)
    array = np.load(root / "d" / f"layer_{layer:02d}.npy", mmap_mode="r", allow_pickle=False)
    if array.ndim != 2 or len(array) != original_count:
        raise ValueError("The repair array must have one row per d_index.jsonl record")
    print(f"Layer {layer}: {len(records)} originals in 40 cells; CPU analysis.")
    directions = construct_cell_directions(array, records)
    result = decompose_variance(directions)
    summary = {
        "layer": layer,
        "layer_source": "explicit" if args.layer is not None else "selected_layer.json",
        "limit_per_cell": args.limit,
        "full_source": args.limit is None and len(records) == 12000,
        "originals": len(records),
        "cells": len(LANGUAGES) * len(MISUSES),
        "originals_per_cell": len(records) // (len(LANGUAGES) * len(MISUSES)),
        "hidden_size": array.shape[1],
        "input_space": "unit_directions",
        **result,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Category main effect: {result['category_percent']:.2f}%")
    print(f"Language main effect: {result['language_percent']:.2f}%")
    print(f"Saved {output / 'summary.json'}")


if __name__ == "__main__":
    main()
