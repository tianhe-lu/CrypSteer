import argparse
import json
from pathlib import Path

import numpy as np
import torch
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
    cells = {(language, misuse): [] for language in LANGUAGES for misuse in MISUSES}
    for record in records:
        cell = record["language"], record["misuse_id"]
        if cell not in cells:
            raise ValueError(f"Unknown language or misuse in {record['original_id']}")
        cells[cell].append(record)
    selected = []
    for cell, members in cells.items():
        if not members or (limit is not None and len(members) < limit):
            raise ValueError(f"{cell} has {len(members)} originals; all 40 cells are required")
        selected.extend(members[:limit])
    selected.sort(key=lambda r: (r["language"], r["misuse_id"], r["task_id"], r["original_id"]))
    return selected, len(records)


def unit_direction(vector, epsilon):
    tensor = torch.from_numpy(np.asarray(vector, dtype=np.float32))
    norm = tensor.norm()
    if not torch.isfinite(norm) or float(norm) <= epsilon:
        raise ValueError("Cannot construct a unit direction from a zero or non-finite vector")
    return (tensor / norm).numpy()


def candidate_rank(candidate):
    return (candidate["positive_categories"], candidate["minimum_category_mean"],
            candidate["mean_category_alignment"], candidate["positive_cells"],
            candidate["minimum_cell_mean"], candidate["explained_energy_ratio"],
            -candidate["pc"], candidate["sign"])


def select_component(u, components, energy_ratios, records, progress=None):
    categories = np.asarray([r["misuse_id"] for r in records])
    cell_labels = np.asarray([f"{r['language']}/{r['misuse_id']}" for r in records])
    category_masks = {g: categories == g for g in sorted(set(categories))}
    cell_masks = {c: cell_labels == c for c in sorted(set(cell_labels))}
    u64 = u.astype(np.float64)
    candidates = []
    for index, component in enumerate(components):
        projections = u64 @ component.astype(np.float64)
        for sign in (1, -1):
            signed = float(sign) * projections
            category_means = {g: float(signed[mask].mean(dtype=np.float64))
                              for g, mask in category_masks.items()}
            cell_means = [float(signed[mask].mean(dtype=np.float64)) for mask in cell_masks.values()]
            values = np.asarray(list(category_means.values()), dtype=np.float64)
            candidates.append({
                "pc": index + 1, "sign": sign,
                "positive_categories": int(np.sum(values > 0)),
                "minimum_category_mean": float(values.min()),
                "mean_category_alignment": float(values.mean()),
                "positive_cells": sum(value > 0 for value in cell_means),
                "minimum_cell_mean": min(cell_means),
                "explained_energy_ratio": float(energy_ratios[index]),
                "category_means": category_means,
            })
        if progress is not None:
            progress.update(1)
    return max(candidates, key=candidate_rank), candidates


def fit_directions(d, records, config, show_progress=True):
    if d.ndim != 2 or d.shape[0] != len(records) or not np.isfinite(d).all():
        raise ValueError("Repair vectors must be a finite matrix matching the selected index")
    d = np.asarray(d, dtype=np.float32)
    epsilon = config["normalization_epsilon"]
    norms = np.linalg.norm(d, axis=1, keepdims=True)
    if np.any(norms <= epsilon):
        raise ValueError("CrypSteer cannot be fitted with zero repair vectors")
    u = (d / norms).astype(np.float32, copy=False)
    rank = min(config["candidate_components"], *d.shape)
    if rank < 1:
        raise ValueError("No SVD components available")
    mean_order = sorted(range(len(records)), key=lambda i: (
        LANGUAGES.index(records[i]["language"]), records[i]["misuse_id"],
        records[i]["task_id"], records[i]["original_id"]))
    cell_index, cell_directions = [], []
    with tqdm(total=43 + rank, desc="Fit directions", unit="step", disable=not show_progress) as progress:
        for language in LANGUAGES:
            for misuse in MISUSES:
                positions = [i for i, r in enumerate(records)
                             if r["language"] == language and r["misuse_id"] == misuse]
                if not positions:
                    raise ValueError(f"Missing cell: {language}/{misuse}")
                cell_directions.append(unit_direction(d[positions].mean(axis=0), epsilon))
                cell_index.append({"row": len(cell_index), "language": language,
                                   "misuse_id": misuse, "originals": len(positions)})
                progress.update(1)
        v_pooled = unit_direction(d[mean_order].mean(axis=0), epsilon)
        v_norm = unit_direction(u[mean_order].mean(axis=0), epsilon)
        progress.update(2)
        progress.set_description("Fit directions: exact SVD")
        _, singular_values, vh = np.linalg.svd(u, full_matrices=False)
        components = vh[:rank].astype(np.float32, copy=True)
        singular_values = singular_values[:rank].astype(np.float32)
        del _, vh
        total_energy = 0.0
        for start in range(0, len(u), 2048):
            block = u[start:start + 2048]
            total_energy += float(np.einsum("ij,ij->", block, block, dtype=np.float64))
        energy_ratios = (np.square(singular_values.astype(np.float64)) / total_energy).astype(np.float32)
        raw_mean = d.mean(axis=0)
        orientation = []
        for component in components:
            dot = float(np.dot(component, raw_mean))
            if abs(dot) > config["orientation_epsilon"]:
                if dot < 0:
                    component *= -1
                orientation.append("raw_repair_mean")
            else:
                projections = u @ component
                if projections[np.argmax(np.abs(projections))] < 0:
                    component *= -1
                orientation.append("strongest_repair_projection")
        progress.update(1)
        progress.set_description("Fit directions: select PC")
        selected, candidates = select_component(u, components, energy_ratios, records, progress)
    v_CrypSteer = unit_direction(np.float32(selected["sign"]) * components[selected["pc"] - 1], epsilon)
    directions = {"v_cell": np.stack(cell_directions), "v_pooled": v_pooled,
                  "v_norm": v_norm, "v_CrypSteer": v_CrypSteer}
    svd = {"components": components, "singular_values": singular_values,
           "explained_energy_ratio": energy_ratios}
    selection = {"selected_pc": selected["pc"], "selected_sign": selected["sign"],
                 "candidate_components": rank, "basis_orientation": orientation,
                 "ranking": ["positive_categories", "minimum_category_mean", "mean_category_alignment",
                             "positive_cells", "minimum_cell_mean", "explained_energy_ratio",
                             "lower_pc_index", "positive_sign"],
                 "selected_scores": selected, "candidates": candidates}
    return directions, cell_index, svd, selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Model directory containing repair_vectors/ and layer_probes/")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config/directions.json")
    parser.add_argument("--output-dir", type=Path, help="Model directory for directions/; defaults to --input-dir")
    parser.add_argument("--layer", type=int, help="Explicit zero-indexed layer; otherwise use selected_layer.json")
    parser.add_argument("--limit", type=int, help="Use the first N originals per cell (40*N in total)")
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if type(config["candidate_components"]) is not int or config["candidate_components"] < 1:
        raise ValueError("candidate_components must be a positive integer")
    for name in ("normalization_epsilon", "orientation_epsilon"):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    layer = args.layer
    if layer is None:
        selection_path = args.input_dir / "layer_probes/selected_layer.json"
        if not selection_path.is_file():
            raise FileNotFoundError("Finish step 03 first, or use --layer for an explicit-layer test")
        with selection_path.open(encoding="utf-8") as handle:
            layer = json.load(handle)["selected_layer"]
    if type(layer) is not int or layer < 0:
        raise ValueError("Layer must be a nonnegative integer")
    root = args.input_dir / "repair_vectors"
    records, input_count = load_originals(root / "d_index.jsonl", args.limit)
    array = np.load(root / "d" / f"layer_{layer:02d}.npy", mmap_mode="r", allow_pickle=False)
    if array.ndim != 2 or array.shape[0] != input_count or array.shape[1] == 0 or array.dtype != np.float32:
        raise ValueError("Expected float32 repair vectors with one row per indexed original")
    d = np.ascontiguousarray(array[[r["row"] for r in records]], dtype=np.float32)
    del array
    output = (args.output_dir if args.output_dir is not None else args.input_dir) / "directions"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Results already exist in {output}; choose another --output-dir")
    print(f"Using {len(records)} originals at layer {layer}; exact SVD runs on the CPU.")
    directions, cell_index, svd, selection = fit_directions(d, records, config)
    output.mkdir(parents=True, exist_ok=True)
    for name, values in directions.items():
        np.save(output / f"{name}.npy", values, allow_pickle=False)
    np.savez(output / "svd.npz", **svd)
    with (output / "cell_index.jsonl").open("w", encoding="utf-8") as handle:
        for row in cell_index:
            handle.write(json.dumps(row) + "\n")
    selection.update({"layer": layer, "layer_source": "explicit" if args.layer is not None else "selected_layer.json",
                      "originals": len(records), "cells": len(cell_index), "hidden_size": d.shape[1],
                      "full_source": args.limit is None and all(r["originals"] == 300 for r in cell_index),
                      "limit_per_cell": args.limit, "config": config})
    with (output / "selection.json").open("w", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"CrypSteer: PC{selection['selected_pc']}, sign {selection['selected_sign']:+d}")
    print(f"Saved 40 Cell DiffMean and three shared unit directions to {output}")


if __name__ == "__main__":
    main()
