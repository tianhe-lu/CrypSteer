"""Cluster unit cell repair directions and compare with category and language labels."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm


LANGUAGES = ("java", "typescript", "csharp", "python")
MISUSES = tuple(f"M{i:02d}" for i in range(1, 11))


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
    selected, counts = [], []
    for cell, members in groups.items():
        members.sort(key=lambda r: (r["task_id"], r["sample_id"]))
        if not members or (limit is not None and len(members) < limit):
            raise ValueError(f"{cell} has {len(members)} originals; all 40 cells are required")
        selected.extend(members[:limit])
        counts.append(len(members[:limit]))
    if len(set(counts)) != 1:
        raise ValueError("Use the same number of originals in every language/category cell")
    return selected, len(records)


def construct_cell_directions(array, records, epsilon=1e-12, show_progress=True):
    if array.ndim != 2 or array.dtype != np.float32 or not array.shape[1]:
        raise ValueError("Expected a float32 matrix of original-level repair vectors d")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("normalization_epsilon must be finite and positive")
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
        means.append(d.mean(axis=0))
    # The original clustering averages raw d, then normalizes cell means in
    # float32. Keep this precision through K-means fitting.
    matrix = np.asarray(means, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= epsilon):
        raise ValueError("Cell directions require finite, nonzero mean repair vectors")
    return list(groups), (matrix / norms).astype(np.float32, copy=False)


def canonicalize_clusters(raw, labels):
    values = np.asarray(raw, dtype=np.int32)
    if values.ndim != 1 or len(values) != len(labels):
        raise ValueError("Cluster assignments and cell labels must be aligned")
    old_ids = sorted(set(values.tolist()), key=lambda cluster: tuple(
        sorted(labels[index] for index in np.flatnonzero(values == cluster))))
    mapping = {old: new for new, old in enumerate(old_ids)}
    return np.asarray([mapping[int(value)] for value in values], dtype=np.int32)


def cluster_directions(directions, cells, config):
    expected_cells = [(language, misuse) for language in LANGUAGES for misuse in MISUSES]
    if (directions.ndim != 2 or directions.dtype != np.float32 or not directions.shape[1]
            or directions.shape[0] != len(expected_cells) or list(cells) != expected_cells
            or not np.isfinite(directions).all()):
        raise ValueError("Expected 40 float32 cell directions in language/category order")
    cluster_count = config["cluster_count"]
    if type(cluster_count) is not int or not 2 <= cluster_count < len(cells):
        raise ValueError("cluster_count must be an integer between 2 and 39")
    fitted = KMeans(n_clusters=cluster_count, random_state=config["random_seed"],
                    n_init=config["n_init"], max_iter=config["max_iter"],
                    tol=config["tolerance"]).fit(directions)
    # Renumber clusters by their members, without changing the partition.
    assignments = canonicalize_clusters(fitted.labels_, [f"{l}:{g}" for l, g in cells])
    if len(set(assignments.tolist())) != cluster_count:
        raise RuntimeError(f"K-means returned fewer than {cluster_count} nonempty clusters")
    language_target = np.asarray([LANGUAGES.index(l) for l, _ in cells], dtype=np.int32)
    category_target = np.asarray([MISUSES.index(g) for _, g in cells], dtype=np.int32)
    return assignments, {
        "category_ari": float(adjusted_rand_score(category_target, assignments)),
        "language_ari": float(adjusted_rand_score(language_target, assignments)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Model directory containing repair_vectors/ and layer_probes/")
    parser.add_argument("--output-dir", type=Path,
                        help="Model directory for clustering/; defaults to --input-dir")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config/clustering.json")
    parser.add_argument("--layer", type=int, help="Zero-indexed layer; otherwise read step 03's selected layer")
    parser.add_argument("--limit", type=int, help="First N originals per cell, ordered by task and sample (40*N total)")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    layer = args.layer
    if layer is None:
        path = args.input_dir / "layer_probes/selected_layer.json"
        if not path.is_file():
            raise FileNotFoundError("Finish step 03 first, or provide --layer")
        layer = json.loads(path.read_text(encoding="utf-8"))["selected_layer"]
    if type(layer) is not int or layer < 0:
        raise ValueError("Layer must be a nonnegative integer")
    output = (args.output_dir if args.output_dir is not None else args.input_dir) / "clustering"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Results already exist in {output}; choose another --output-dir")
    root = args.input_dir / "repair_vectors"
    records, original_count = load_originals(root, args.limit)
    array = np.load(root / "d" / f"layer_{layer:02d}.npy", mmap_mode="r", allow_pickle=False)
    if array.ndim != 2 or len(array) != original_count:
        raise ValueError("The repair array must have one row per d_index.jsonl record")
    print(f"Layer {layer}: {len(records)} originals in 40 cells; CPU clustering.")
    cells, directions = construct_cell_directions(array, records, config["normalization_epsilon"])
    print(f"K-means: {config['cluster_count']} clusters, {config['n_init']} initializations.")
    assignments, metrics = cluster_directions(directions, cells, config)
    originals_per_cell = len(records) // len(cells)
    summary = {
        "layer": layer,
        "layer_source": "explicit" if args.layer is not None else "selected_layer.json",
        "limit_per_cell": args.limit,
        "full_source": args.limit is None and len(records) == 12000,
        "originals": len(records),
        "cells": len(cells),
        "originals_per_cell": originals_per_cell,
        "hidden_size": array.shape[1],
        "dtype": "float32",
        "config": config,
        **metrics,
    }
    output.mkdir(parents=True, exist_ok=True)
    # Save the exact float32 unit directions used for clustering. Array row i
    # corresponds to data row i in assignments.csv (excluding its header).
    np.save(output / "cell_directions.npy", directions, allow_pickle=False)
    with (output / "assignments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["language", "misuse_id", "originals", "cluster"])
        writer.writeheader()
        writer.writerows({"language": language, "misuse_id": misuse,
                          "originals": originals_per_cell, "cluster": int(cluster)}
                         for (language, misuse), cluster in zip(cells, assignments))
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Category ARI: {metrics['category_ari']:.6f}; language ARI: {metrics['language_ari']:.6f}")
    print(f"Saved cell_directions.npy, assignments.csv, and summary.json to {output}")


if __name__ == "__main__":
    main()
