import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


LANGUAGES = ("java", "typescript", "csharp", "python")
MISUSES = tuple(f"M{i:02d}" for i in range(1, 11))
IDENTITY_FIELDS = ("original_id", "language", "misuse_id", "task_id", "sample_id")


def identity(record):
    return tuple(record[field] for field in ("language", "misuse_id", "task_id", "sample_id"))


def load_originals(path, limit):
    if limit is not None and (type(limit) is not int or limit < 2):
        raise ValueError("--limit must be at least 2 originals per language/category/task stratum")
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    if not records or len({r["original_id"] for r in records}) != len(records):
        raise ValueError("Expected one index record per original")
    rows = [r["row"] for r in records]
    if any(type(row) is not int for row in rows) or sorted(rows) != list(range(len(records))):
        raise ValueError("Repair index rows must cover the array without gaps")
    groups = defaultdict(list)
    for record in records:
        if record["language"] not in LANGUAGES or record["misuse_id"] not in MISUSES:
            raise ValueError(f"Unknown language/category in {record['original_id']}")
        prefix = f"{record['language']}__{record['misuse_id']}_{record['task_id']}_"
        if not record["original_id"].startswith(prefix):
            raise ValueError(f"Original ID does not match its metadata: {record['original_id']}")
        record["sample_id"] = record["original_id"][len(prefix):]
        if (not record["sample_id"] or len(record["pair_rows"]) != 4
                or len(set(record["pair_rows"])) != 4):
            raise ValueError("Each original must identify a sample and four distinct structure pairs")
        groups[(record["language"], record["misuse_id"], record["task_id"])].append(record)
    cells = {(r["language"], r["misuse_id"]) for r in records}
    if cells != {(language, misuse) for language in LANGUAGES for misuse in MISUSES}:
        raise ValueError("All 40 language/category cells are required")
    selected = []
    for stratum, members in sorted(groups.items()):
        members.sort(key=identity)
        if len(members) < 2 or (limit is not None and len(members) < limit):
            raise ValueError(f"{stratum} has only {len(members)} originals")
        selected.extend(members[:limit])
    return selected, len(records)


def stratified_split(records, discovery_fraction, split_seed, show_progress=True):
    if not 0 < discovery_fraction < 1:
        raise ValueError("discovery_fraction must be between zero and one")
    if type(split_seed) is not int or split_seed < 0:
        raise ValueError("split_seed must be a nonnegative integer")
    if not records or len({identity(r) for r in records}) != len(records):
        raise ValueError("Original identities must be nonempty and unique")
    groups = defaultdict(list)
    for position, record in enumerate(records):
        groups[identity(record)[:3]].append(position)
    discovery = np.zeros(len(records), dtype=bool)
    for stratum, positions in tqdm(sorted(groups.items()), desc="Split originals", unit="stratum",
                                   disable=not show_progress):
        positions.sort(key=lambda i: identity(records[i]))
        count = int(round(len(positions) * discovery_fraction))
        if count < 1 or count >= len(positions):
            raise ValueError(f"Stratum {stratum} cannot populate both discovery and test")

        text = "::".join([str(split_seed), "pca-heldout", *stratum])
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")
        chosen = np.random.default_rng(seed).permutation(len(positions))[:count]
        discovery[np.asarray(positions, dtype=int)[chosen]] = True
    return discovery


def candidate_rank(candidate):
    return (candidate["positive_categories"], candidate["minimum_category_mean"],
            candidate["mean_category_alignment"], candidate["positive_cells"],
            candidate["minimum_cell_mean"], candidate["explained_energy_ratio"],
            -candidate["pc"], candidate["sign"])


def fit_discovery_direction(d, records, config, show_progress=True):
    d = np.asarray(d, dtype=np.float32)
    if d.ndim != 2 or not len(d) or not d.shape[1] or len(d) != len(records) or not np.isfinite(d).all():
        raise ValueError("Discovery repairs must be a finite matrix matching the discovery index")
    norms = np.linalg.norm(d, axis=1, keepdims=True)
    if np.any(norms <= config["normalization_epsilon"]):
        raise ValueError("Discovery repairs cannot contain zero or near-zero vectors")
    u = (d / norms).astype(np.float32, copy=False)
    rank = min(config["candidate_components"], *d.shape)
    if rank < 1:
        raise ValueError("candidate_components must be positive")
    with tqdm(total=1 + rank, desc="Discovery: exact SVD", unit="step",
              disable=not show_progress) as progress:
        left, singular_values, vh = np.linalg.svd(u, full_matrices=False)
        components = vh[:rank].astype(np.float32, copy=True)
        energy = float(np.einsum("ij,ij->", u, u, dtype=np.float64))
        energy_ratios = np.square(singular_values[:rank].astype(np.float64)) / energy
        del left, vh
        progress.update(1)
        progress.set_description("Discovery: select PC")
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
                cell_means = [float(signed[mask].mean(dtype=np.float64))
                              for mask in cell_masks.values()]
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
            progress.update(1)
    selected = max(candidates, key=candidate_rank)

    v_disc = (np.float32(selected["sign"]) * components[selected["pc"] - 1]).astype(np.float32)
    selection = {"selected_pc": selected["pc"], "selected_sign": selected["sign"],
                 "candidate_components": rank,
                 "ranking": ["positive_categories", "minimum_category_mean", "mean_category_alignment",
                             "positive_cells", "minimum_cell_mean", "explained_energy_ratio",
                             "lower_pc_index", "positive_sign"],
                 "selected_scores": selected, "candidates": candidates}
    return v_disc, selection


def analyze_alignment(d, records, discovery, v_disc, show_progress=True):
    d = np.asarray(d)
    discovery = np.asarray(discovery)
    v_disc = np.asarray(v_disc)
    if (d.ndim != 2 or len(d) != len(records) or not np.isfinite(d).all()
            or discovery.dtype != np.bool_ or discovery.shape != (len(records),)
            or not discovery.any() or discovery.all()):
        raise ValueError("Repairs and discovery membership must cover both split sides")
    if (v_disc.shape != (d.shape[1],) or not np.isfinite(v_disc).all()
            or not np.isclose(np.linalg.norm(v_disc), 1, rtol=1e-5, atol=1e-6)):
        raise ValueError("The discovery direction must be a finite unit vector of the same width")
    positions = np.flatnonzero(~discovery)
    test_d = d[positions].astype(np.float64)
    norms = np.linalg.norm(test_d, axis=1)
    if np.any(norms <= 0):
        raise ValueError("Test repairs cannot contain zero vectors")
    projections = test_d @ v_disc.astype(np.float64)
    cosines = projections / norms
    alignment = []
    for i, position in enumerate(tqdm(positions, desc="Project test repairs", unit="original",
                                      disable=not show_progress)):
        alignment.append({**{field: records[position][field] for field in IDENTITY_FIELDS},
                          "repair_norm": float(norms[i]), "projection": float(projections[i]),
                          "cosine": float(cosines[i]), "positive": bool(projections[i] > 0)})
    all_cells = [(r["language"], r["misuse_id"]) for r in records]
    test_cells = [all_cells[position] for position in positions]
    cells = []
    for language, misuse in sorted(set(all_cells)):
        mask = np.asarray([cell == (language, misuse) for cell in test_cells])
        count = int(mask.sum())
        if not count:
            raise ValueError(f"No test originals in {language}/{misuse}")
        positive_count = int(np.sum(projections[mask] > 0))
        cells.append({"language": language, "misuse_id": misuse,
                      "discovery_originals": int(sum(is_discovery and cell == (language, misuse)
                                                     for cell, is_discovery in zip(all_cells, discovery))),
                      "test_originals": count, "positive_count": positive_count,
                      "positive_percent": 100.0 * positive_count / count,
                      "mean_cosine": float(cosines[mask].mean(dtype=np.float64))})
    positive_count = int(np.sum(projections > 0))
    summary = {"originals": len(records), "discovery_originals": int(discovery.sum()),
               "test_originals": len(positions), "positive_count": positive_count,
               "positive_percent": 100.0 * positive_count / len(positions),
               "mean_cosine": float(cosines.mean(dtype=np.float64)),
               "positive_cells": sum(cell["mean_cosine"] > 0 for cell in cells), "cells": len(cells)}
    return alignment, cells, summary


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Model directory containing repair_vectors/ and layer_probes/")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config/heldout.json")
    parser.add_argument("--output-dir", type=Path,
                        help="Model directory for heldout_alignment/; defaults to --input-dir")
    parser.add_argument("--layer", type=int, help="Zero-indexed layer; otherwise read step 03's selected layer")
    parser.add_argument("--limit", type=int,
                        help="First N originals per language/category/task stratum, before splitting (N >= 2)")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not 0 < config["discovery_fraction"] < 1:
        raise ValueError("discovery_fraction must be between zero and one")
    for name in ("split_seed", "candidate_components"):
        if type(config[name]) is not int or config[name] < (0 if name == "split_seed" else 1):
            raise ValueError(f"Invalid {name}")
    if not np.isfinite(config["normalization_epsilon"]) or config["normalization_epsilon"] <= 0:
        raise ValueError("normalization_epsilon must be finite and positive")
    layer = args.layer
    if layer is None:
        path = args.input_dir / "layer_probes/selected_layer.json"
        if not path.is_file():
            raise FileNotFoundError("Finish step 03 first, or provide --layer")
        layer = json.loads(path.read_text(encoding="utf-8"))["selected_layer"]
    if type(layer) is not int or layer < 0:
        raise ValueError("Layer must be a nonnegative integer")
    output = (args.output_dir if args.output_dir is not None else args.input_dir) / "heldout_alignment"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Results already exist in {output}; choose another --output-dir")
    records, input_count = load_originals(args.input_dir / "repair_vectors/d_index.jsonl", args.limit)
    array = np.load(args.input_dir / "repair_vectors/d" / f"layer_{layer:02d}.npy",
                    mmap_mode="r", allow_pickle=False)
    if array.ndim != 2 or array.shape[0] != input_count or array.shape[1] == 0 or array.dtype != np.float32:
        raise ValueError("Expected float32 repair vectors with one row per indexed original")
    d = np.ascontiguousarray(array[[r["row"] for r in records]], dtype=np.float32)
    del array
    if not np.isfinite(d).all() or np.any(np.linalg.norm(d, axis=1) <= config["normalization_epsilon"]):
        raise ValueError("Repair vectors must be finite and nonzero")
    discovery = stratified_split(records, config["discovery_fraction"], config["split_seed"])
    print(f"Layer {layer}: {int(discovery.sum())} discovery / {int((~discovery).sum())} test originals.")
    discovery_records = [r for r, selected in zip(records, discovery) if selected]
    v_disc, selection = fit_discovery_direction(d[discovery], discovery_records, config)
    alignment, cells, summary = analyze_alignment(d, records, discovery, v_disc)
    summary.update({"layer": layer,
                    "layer_source": "explicit" if args.layer is not None else "selected_layer.json",
                    "input_originals": input_count, "hidden_size": d.shape[1],
                    "full_population": args.limit is None and len(records) == 12000
                        and all(c["discovery_originals"] + c["test_originals"] == 300 for c in cells),
                    "limit_per_stratum": args.limit, "config": config, "selection": selection})
    output.mkdir(parents=True, exist_ok=True)
    with (output / "split.jsonl").open("w", encoding="utf-8") as handle:
        for record, selected in zip(records, discovery):
            row = {**{field: record[field] for field in IDENTITY_FIELDS}, "row": record["row"],
                   "split": "discovery" if selected else "test"}
            handle.write(json.dumps(row) + "\n")
    np.save(output / "v_CrypSteer_disc.npy", v_disc, allow_pickle=False)
    write_csv(output / "alignment.csv", alignment)
    write_csv(output / "by_cell.csv", cells)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Discovery PC{selection['selected_pc']}, sign {selection['selected_sign']:+d}; "
          f"positive test repairs: {summary['positive_percent']:.2f}%; "
          f"mean cosine: {summary['mean_cosine']:.4f}; "
          f"positive cells: {summary['positive_cells']}/{summary['cells']}.")
    print(f"Saved held-out alignment to {output}")


if __name__ == "__main__":
    main()
