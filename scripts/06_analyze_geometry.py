import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


LANGUAGES = ("java", "typescript", "csharp", "python")
MISUSES = tuple(f"M{i:02d}" for i in range(1, 11))


def load_originals(path, limit):
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1")
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    if not records or len({r["original_id"] for r in records}) != len(records):
        raise ValueError("Expected one index record per original")
    rows = [r["row"] for r in records]
    if any(type(row) is not int for row in rows) or sorted(rows) != list(range(len(records))):
        raise ValueError("Repair index rows must cover the array without gaps")
    groups = {(language, misuse): [] for language in LANGUAGES for misuse in MISUSES}
    for record in records:
        cell = record["language"], record["misuse_id"]
        if cell not in groups:
            raise ValueError(f"Unknown language or misuse in {record['original_id']}")
        groups[cell].append(record)
    selected = []
    for cell, members in groups.items():
        if not members or (limit is not None and len(members) < limit):
            raise ValueError(f"{cell} has {len(members)} originals; all 40 cells are required")
        selected.extend(members[:limit])
    selected.sort(key=lambda r: (r["misuse_id"], LANGUAGES.index(r["language"]),
                                 r["task_id"], r["original_id"]))
    return selected, len(records)


def unit_rows(values):
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise ValueError("Cosine requires nonzero, finite cell means and reference directions")
    return values / norms


def analyze_geometry(d, records, show_progress=True):
    if d.ndim != 2 or d.shape[0] != len(records) or d.shape[1] == 0 or not np.isfinite(d).all():
        raise ValueError("Repair vectors must be a finite matrix matching the selected index")
    d = np.asarray(d, dtype=np.float32)
    groups = {(language, misuse): [] for misuse in MISUSES for language in LANGUAGES}
    for position, record in enumerate(records):
        groups[record["language"], record["misuse_id"]].append(position)
    cells, cell_means = [], []
    with tqdm(total=80, desc="Repair geometry", unit="cell", disable=not show_progress) as progress:
        for (language, misuse), positions in groups.items():
            if not positions:
                raise ValueError(f"Missing cell: {language}/{misuse}")
            block = d[positions]
            cell_means.append(block.mean(axis=0))
            mean_norm = float(np.linalg.norm(block.astype(np.float64), axis=1).mean())
            cells.append({"row": len(cells), "language": language, "misuse_id": misuse,
                          "originals": len(positions), "mean_original_norm": mean_norm})
            progress.update(1)

        v_cell = unit_rows(np.stack(cell_means))
        cosine = np.clip(v_cell @ v_cell.T, -1.0, 1.0)
        rho = np.asarray([cell["mean_original_norm"] for cell in cells], dtype=np.float64)
        norm_ratio = rho[:, None] / rho[None, :]
        grid = v_cell.reshape(len(MISUSES), len(LANGUAGES), -1)
        for language_index in range(len(LANGUAGES)):
            references = unit_rows(np.delete(grid, language_index, axis=1).mean(axis=1))
            scores = grid[:, language_index, :] @ references.T
            predicted = np.argsort(-scores, axis=1, kind="stable")[:, 0]
            for category_index, prediction in enumerate(predicted):
                cell = cells[category_index * len(LANGUAGES) + language_index]
                cell["predicted_misuse_id"] = MISUSES[int(prediction)]
                cell["correct"] = bool(prediction == category_index)
                progress.update(1)

    languages = np.asarray([cell["language"] for cell in cells])
    categories = np.asarray([cell["misuse_id"] for cell in cells])
    upper = np.triu(np.ones(cosine.shape, dtype=bool), k=1)
    same_category = categories[:, None] == categories[None, :]
    same_language = languages[:, None] == languages[None, :]
    across_languages = upper & same_category & ~same_language
    within_language = upper & same_language & ~same_category
    summary = {
        "originals": len(records), "cells": len(cells),
        "same_category_across_languages_cosine": float(cosine[across_languages].mean()),
        "same_category_across_languages_pairs": int(across_languages.sum()),
        "different_categories_within_language_cosine": float(cosine[within_language].mean()),
        "different_categories_within_language_pairs": int(within_language.sum()),
        "max_min_mean_original_norm_ratio": float(rho.max() / rho.min()),
        "retrieval_correct": sum(cell["correct"] for cell in cells),
        "retrieval_total": len(cells),
    }
    return {"cosine": cosine, "norm_ratio": norm_ratio}, cells, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Model directory containing repair_vectors/ and layer_probes/")
    parser.add_argument("--output-dir", type=Path, help="Model directory for geometry/; defaults to --input-dir")
    parser.add_argument("--layer", type=int, help="Explicit zero-indexed layer; otherwise use selected_layer.json")
    parser.add_argument("--limit", type=int, help="Use the first N originals per cell (40*N in total)")
    args = parser.parse_args()
    layer = args.layer
    if layer is None:
        selection_path = args.input_dir / "layer_probes/selected_layer.json"
        if not selection_path.is_file():
            raise FileNotFoundError("Finish step 03 first, or use --layer for an explicit-layer test")
        with selection_path.open(encoding="utf-8") as handle:
            layer = json.load(handle)["selected_layer"]
    if type(layer) is not int or layer < 0:
        raise ValueError("Layer must be a nonnegative integer")
    output = (args.output_dir if args.output_dir is not None else args.input_dir) / "geometry"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Results already exist in {output}; choose another --output-dir")
    root = args.input_dir / "repair_vectors"
    records, input_count = load_originals(root / "d_index.jsonl", args.limit)
    array = np.load(root / "d" / f"layer_{layer:02d}.npy", mmap_mode="r", allow_pickle=False)
    if array.ndim != 2 or array.shape[0] != input_count or array.shape[1] == 0 or array.dtype != np.float32:
        raise ValueError("Expected float32 repair vectors with one row per indexed original")
    d = np.ascontiguousarray(array[[r["row"] for r in records]], dtype=np.float32)
    del array
    print(f"Using {len(records)} originals at layer {layer}; geometry analysis runs on the CPU.")
    matrices, cells, summary = analyze_geometry(d, records)
    summary.update({"layer": layer,
                    "layer_source": "explicit" if args.layer is not None else "selected_layer.json",
                    "limit_per_cell": args.limit,
                    "full_source": args.limit is None and all(cell["originals"] == 300 for cell in cells)})
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "geometry.npz", **matrices)
    with (output / "cells.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Same-category cosine across languages: {summary['same_category_across_languages_cosine']:.6f}")
    print(f"Different-category cosine within a language: {summary['different_categories_within_language_cosine']:.6f}")
    print(f"Largest/smallest mean repair length: {summary['max_min_mean_original_norm_ratio']:.6f}")
    print(f"Cross-language category retrieval: {summary['retrieval_correct']}/{summary['retrieval_total']}")
    print(f"Saved geometry.npz, cells.csv, and summary.json to {output}")


if __name__ == "__main__":
    main()
