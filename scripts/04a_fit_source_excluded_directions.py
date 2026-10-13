import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


LANGUAGES = ("java", "typescript", "csharp", "python")
MISUSES = tuple(f"M{i:02d}" for i in range(1, 11))
SCOPES = ("language", "misuse", "both")
RANKING = ("positive_categories", "minimum_category_mean", "mean_category_alignment",
           "positive_cells", "minimum_cell_mean", "explained_energy_ratio",
           "lower_pc_index", "positive_sign")


def load_originals(path, limit=None):
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
    selected.sort(key=lambda r: (LANGUAGES.index(r["language"]), r["misuse_id"],
                                 r["task_id"], r["original_id"]))
    return selected, len(records)


def exclusion_targets(scope):
    if scope == "language":
        return [(language, None) for language in LANGUAGES]
    if scope == "misuse":
        return [(None, misuse) for misuse in MISUSES]
    if scope == "both":
        return [(language, misuse) for language in LANGUAGES for misuse in MISUSES]
    raise ValueError(f"Unknown exclusion scope: {scope}")


def retained_records(records, scope, target_language, target_misuse):
    if (target_language, target_misuse) not in exclusion_targets(scope):
        raise ValueError(f"Invalid target for {scope}: {(target_language, target_misuse)}")
    return [r for r in records
            if (target_language is None or r["language"] != target_language)
            and (target_misuse is None or r["misuse_id"] != target_misuse)]


def candidate_rank(candidate):
    return (candidate["positive_categories"], candidate["minimum_category_mean"],
            candidate["mean_category_alignment"], candidate["positive_cells"],
            candidate["minimum_cell_mean"], candidate["explained_energy_ratio"],
            -candidate["pc"], candidate["sign"])


def fit_direction(d, records, config):
    d = np.asarray(d, dtype=np.float32)
    if d.ndim != 2 or not len(d) or not d.shape[1] or d.shape[0] != len(records) or not np.isfinite(d).all():
        raise ValueError("Repair vectors must be a nonempty finite matrix matching the index")
    epsilon = config["normalization_epsilon"]
    if type(config["candidate_components"]) is not int or config["candidate_components"] < 1:
        raise ValueError("candidate_components must be a positive integer")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("normalization_epsilon must be finite and positive")
    norms = np.linalg.norm(d, axis=1, keepdims=True)
    if np.any(norms <= epsilon):
        raise ValueError("CrypSteer cannot be fitted with zero repair vectors")
    u = (d / norms).astype(np.float32, copy=False)
    left, singular_values, vh = np.linalg.svd(u, full_matrices=False)
    rank = min(config["candidate_components"], len(singular_values), len(vh))
    components = vh[:rank].astype(np.float32, copy=True)
    del left, vh
    energy = float(np.einsum("ij,ij->", u, u, dtype=np.float64))
    energy_ratios = np.square(singular_values[:rank].astype(np.float64)) / energy
    categories = np.asarray([r["misuse_id"] for r in records])
    cells = np.asarray([f"{r['language']}/{r['misuse_id']}" for r in records])
    category_masks = {g: categories == g for g in sorted(set(categories))}
    cell_masks = {c: cells == c for c in sorted(set(cells))}
    u64 = u.astype(np.float64)
    candidates = []
    for index, component in enumerate(components):
        projections = u64 @ component.astype(np.float64)
        for sign in (1, -1):
            signed = float(sign) * projections
            category_means = {g: float(signed[mask].mean(dtype=np.float64))
                              for g, mask in category_masks.items()}
            cell_means = {c: float(signed[mask].mean(dtype=np.float64))
                          for c, mask in cell_masks.items()}
            group_values = np.asarray(list(category_means.values()), dtype=np.float64)
            cell_values = np.asarray(list(cell_means.values()), dtype=np.float64)
            candidates.append({
                "pc": index + 1, "sign": sign,
                "positive_categories": int(np.sum(group_values > 0)),
                "minimum_category_mean": float(group_values.min()),
                "mean_category_alignment": float(group_values.mean()),
                "positive_cells": int(np.sum(cell_values > 0)),
                "minimum_cell_mean": float(cell_values.min()),
                "explained_energy_ratio": float(energy_ratios[index]),
                "category_means": category_means, "cell_means": cell_means,
            })
    selected = max(candidates, key=candidate_rank)
    component = np.float32(selected["sign"]) * components[selected["pc"] - 1]
    tensor = torch.from_numpy(component)
    norm = tensor.norm()
    if not torch.isfinite(norm) or float(norm) <= epsilon:
        raise ValueError("Selected component cannot be normalized")
    v_CrypSteer = (tensor / norm).numpy()
    return v_CrypSteer, selected, candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Model directory containing repair_vectors/ and layer_probes/")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config/directions.json")
    parser.add_argument("--output-dir", type=Path,
                        help="Model directory for source_excluded_directions/; defaults to --input-dir")
    parser.add_argument("--layer", type=int, help="Explicit layer; otherwise use selected_layer.json")
    parser.add_argument("--scopes", choices=SCOPES, nargs="+", default=list(SCOPES))
    parser.add_argument("--limit", type=int, help="Use the first N originals per cell before excluding sources")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config = {name: config[name] for name in ("candidate_components", "normalization_epsilon")}
    if type(config["candidate_components"]) is not int or config["candidate_components"] < 1:
        parser.error("candidate_components must be a positive integer")
    if not np.isfinite(config["normalization_epsilon"]) or config["normalization_epsilon"] <= 0:
        parser.error("normalization_epsilon must be finite and positive")
    layer = args.layer
    if layer is None:
        selected_path = args.input_dir / "layer_probes/selected_layer.json"
        if not selected_path.is_file():
            raise FileNotFoundError("Finish step 03 first, or use --layer for an explicit-layer test")
        layer = json.loads(selected_path.read_text(encoding="utf-8"))["selected_layer"]
    if type(layer) is not int or layer < 0:
        parser.error("Layer must be a nonnegative integer")
    scopes = list(dict.fromkeys(args.scopes))
    output = (args.output_dir or args.input_dir) / "source_excluded_directions"
    for scope in scopes:
        directory = output / f"exclude_{scope}"
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise FileExistsError(f"Results already exist in {directory}; choose another --output-dir")

    root = args.input_dir / "repair_vectors"
    records, input_count = load_originals(root / "d_index.jsonl", args.limit)
    array = np.load(root / "d" / f"layer_{layer:02d}.npy", mmap_mode="r", allow_pickle=False)
    if array.ndim != 2 or array.shape[0] != input_count or not array.shape[1] or array.dtype != np.float32:
        raise ValueError("Expected float32 repair vectors with one row per indexed original")
    cell_counts = {(language, misuse): sum(r["language"] == language and r["misuse_id"] == misuse
                                         for r in records)
                   for language in LANGUAGES for misuse in MISUSES}
    full_retained_source = args.limit is None and all(count == 300 for count in cell_counts.values())
    total = sum(len(exclusion_targets(scope)) for scope in scopes)
    print(f"Using {len(records)} originals at layer {layer}; {total} exact SVD fits on the CPU.")
    with tqdm(total=total, desc="Fit excluded directions", unit="direction") as progress:
        for scope in scopes:
            directions, index = [], []
            for target_language, target_misuse in exclusion_targets(scope):
                target = "/".join(value for value in (target_language, target_misuse) if value is not None)
                progress.set_description(f"Exclude {scope}: {target}")
                retained = retained_records(records, scope, target_language, target_misuse)
                d = np.ascontiguousarray(array[[r["row"] for r in retained]], dtype=np.float32)
                vector, selected, _ = fit_direction(d, retained, config)
                directions.append(vector)
                index.append({"row": len(index), "target_language": target_language,
                              "target_misuse": target_misuse, "originals": len(retained),
                              "cells": len({(r["language"], r["misuse_id"]) for r in retained}),
                              "selected_pc": selected["pc"], "selected_sign": selected["sign"],
                              "selected_scores": selected})
                del d
                progress.set_postfix(pc=selected["pc"], originals=len(retained))
                progress.update(1)
            directory = output / f"exclude_{scope}"
            directory.mkdir(parents=True, exist_ok=True)
            np.save(directory / "v_CrypSteer.npy", np.stack(directions), allow_pickle=False)
            with (directory / "index.jsonl").open("w", encoding="utf-8") as handle:
                for row in index:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
            selection = {"layer": layer, "layer_source": "explicit" if args.layer is not None else "selected_layer.json",
                         "source_scope": f"exclude_{scope}", "hidden_size": array.shape[1],
                         "directions": len(index), "originals": len(records),
                         "limit_per_cell": args.limit, "full_retained_source": full_retained_source,
                         "config": config, "ranking": list(RANKING)}
            (directory / "selection.json").write_text(
                json.dumps(selection, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {total} unit directions to {output}")
    if not full_retained_source:
        print("Subset fit: these directions are for testing.")


if __name__ == "__main__":
    main()
