"""Compare repair geometry across layers using cosine between cell-similarity signatures."""

import argparse
import csv
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


def select_layers(root, layers=None):
    paths = list(root.glob("layer_*.npy"))
    if not paths:
        raise FileNotFoundError(f"No layer arrays in {root}")
    available = sorted(int(path.stem.removeprefix("layer_")) for path in paths)
    if len(set(available)) != len(available) or any(
            not (root / f"layer_{layer:02d}.npy").is_file() for layer in available):
        raise ValueError("Expected one layer_XX.npy file per layer, as saved by step 02")
    if layers is None:
        return available, available
    if (not layers or any(type(layer) is not int or layer < 0 for layer in layers)
            or len(set(layers)) != len(layers)):
        raise ValueError("--layers must contain distinct nonnegative layer indices")
    missing = sorted(set(layers) - set(available))
    if missing:
        raise FileNotFoundError(f"Missing repair arrays for layers {missing}")
    return sorted(layers), available


def cell_signature(array, records, epsilon=EPSILON):
    if array.ndim != 2 or array.dtype != np.float32 or not array.shape[1]:
        raise ValueError("Expected a float32 matrix of original-level repair vectors d")
    groups = {(language, misuse): [] for language in LANGUAGES for misuse in MISUSES}
    for record in records:
        groups[record["language"], record["misuse_id"]].append(record)
    means = []
    for (language, misuse), members in groups.items():
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
    # Keep raw averaging, cell normalization, and matrix multiplication in
    # float32, as in the original cross-layer analysis.
    matrix = np.asarray(means, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(norms).all():
        raise ValueError("Nonfinite cell mean norm")
    directions = (matrix / np.where(norms > epsilon, norms, 1.0)).astype(np.float32, copy=False)
    cosine = directions @ directions.T
    np.fill_diagonal(cosine, 1.0)
    # Exclude the diagonal and avoid counting each cell pair twice.
    return cosine[np.triu_indices(len(groups), k=1)].astype(np.float32, copy=False)


def compute_rsa(signatures, show_progress=True):
    signatures = np.asarray(signatures, dtype=np.float32)
    if (signatures.ndim != 2 or not len(signatures) or signatures.shape[1] != 780
            or not np.isfinite(signatures).all()):
        raise ValueError("Expected 780 finite signed cell similarities per layer")
    result = np.empty((len(signatures), len(signatures)), dtype=np.float32)
    for left, left_signature in enumerate(tqdm(signatures, desc="Compare layers", unit="layer",
                                               disable=not show_progress)):
        for right, right_signature in enumerate(signatures):
            denominator = float(np.linalg.norm(left_signature) * np.linalg.norm(right_signature))
            result[left, right] = (float("nan") if denominator == 0.0
                                  else float(np.dot(left_signature, right_signature) / denominator))
    # Preserve the calculated diagonal and roundoff; do not clip or center.
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Model directory containing step 02's repair_vectors/")
    parser.add_argument("--output-dir", type=Path,
                        help="Model directory for layer_rsa/; defaults to --input-dir")
    parser.add_argument("--layers", type=int, nargs="+", help="Layer indices; defaults to all saved layers")
    parser.add_argument("--limit", type=int, help="First N originals per cell, ordered by task and sample (40*N total)")
    args = parser.parse_args()
    root = args.input_dir / "repair_vectors"
    layers, available = select_layers(root / "d", args.layers)
    records, original_count = load_originals(root, args.limit)
    output = (args.output_dir if args.output_dir is not None else args.input_dir) / "layer_rsa"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Results already exist in {output}; choose another --output-dir")
    print(f"{len(layers)} layers, {len(records)} originals in 40 cells; CPU analysis.")
    signatures, hidden_size = [], None
    for layer in tqdm(layers, desc="Layer geometry", unit="layer"):
        array = np.load(root / "d" / f"layer_{layer:02d}.npy", mmap_mode="r", allow_pickle=False)
        if array.ndim != 2 or len(array) != original_count:
            raise ValueError(f"Layer {layer}: expected one repair row per d_index.jsonl record")
        if hidden_size is None:
            hidden_size = array.shape[1]
        elif array.shape[1] != hidden_size:
            raise ValueError(f"Layer {layer}: hidden width differs from preceding layers")
        signatures.append(cell_signature(array, records))
        del array
    rsa = compute_rsa(np.stack(signatures))
    cell_order = [f"{language}:{misuse}" for language in LANGUAGES for misuse in MISUSES]
    summary = {
        "layers": layers,
        "available_layers": available,
        "all_available_layers": layers == available,
        "limit_per_cell": args.limit,
        "full_source": args.limit is None and len(records) == 12000,
        "originals": len(records),
        "cells": len(cell_order),
        "originals_per_cell": len(records) // len(cell_order),
        "hidden_size": hidden_size,
        "cell_order": cell_order,
        "signature_entries": 780,
        "dtype": "float32",
        "similarity": "cosine",
        "normalization_epsilon": EPSILON,
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "cross_layer_rsa.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", *[f"L{layer}" for layer in layers]])
        for layer, row in zip(layers, rsa):
            writer.writerow([f"L{layer}", *[float(value) for value in row]])
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {len(layers)} x {len(layers)} cross_layer_rsa.csv and summary.json to {output}")


if __name__ == "__main__":
    main()
