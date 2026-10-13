import argparse
import json
from math import sqrt
from pathlib import Path

import numpy as np
from tqdm import tqdm


LANGUAGES = ("java", "typescript", "csharp", "python")
MISUSES = tuple(f"M{i:02d}" for i in range(1, 11))


def load_programs(index_path, limit):
    if limit is not None and limit < 2:
        raise ValueError("--limit must be at least 2 for sample variance")
    with index_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    if len(records) < 2 or len({r["id"] for r in records}) != len(records):
        raise ValueError("Expected at least two programs with unique IDs")
    rows = [r["row"] for r in records]
    if any(type(row) is not int for row in rows) or sorted(rows) != list(range(len(records))):
        raise ValueError("Program index rows must cover the array without gaps")
    if limit is not None and limit > len(records):
        raise ValueError(f"Requested {limit} programs, but found {len(records)}")
    selected = records[:limit]
    for record in selected:
        if record["language"] not in LANGUAGES or record["misuse_id"] not in MISUSES:
            raise ValueError(f"Unexpected language or misuse category in {record['id']}")
        if record["label"] not in {"secure", "misuse"}:
            raise ValueError(f"Unexpected program label in {record['id']}")
    return selected, len(records)


def calibrate_sigma(h_bar, records, show_progress=True):
    if h_bar.ndim != 2 or h_bar.shape[1] == 0 or len(records) < 2:
        raise ValueError("Sigma requires at least two nonempty program representations")
    cells = {(language, misuse): [] for language in LANGUAGES for misuse in MISUSES}
    for record in records:
        cells[record["language"], record["misuse_id"]].append(record)
    N_prog, mu_h, squared_deviation_sum = 0, None, 0.0
    with tqdm(total=len(records), desc="Calibrate sigma", unit="program", disable=not show_progress) as progress:
        for members in cells.values():
            if not members:
                continue
            members.sort(key=lambda r: (r["task_id"], r["original_id"], r["id"]))
            batch = np.asarray(h_bar[[r["row"] for r in members]], dtype=np.float64)
            if not np.isfinite(batch).all():
                raise ValueError("Program representations contain non-finite values")
            batch_count = len(batch)
            batch_mean = batch.mean(axis=0, dtype=np.float64)
            centered = batch - batch_mean
            batch_squared_deviation = float(np.einsum("ij,ij->", centered, centered))
            if mu_h is None:
                mu_h = batch_mean
                squared_deviation_sum = batch_squared_deviation
                N_prog = batch_count
            else:
                combined_count = N_prog + batch_count
                mean_shift = batch_mean - mu_h
                squared_deviation_sum += batch_squared_deviation + float(
                    np.dot(mean_shift, mean_shift) * N_prog * batch_count / combined_count)
                mu_h = mu_h + mean_shift * (batch_count / combined_count)
                N_prog = combined_count
            progress.update(batch_count)
            del batch, centered

    sigma = sqrt(max(squared_deviation_sum, 0.0) / (N_prog - 1))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("The intervention scale must be finite and positive")
    return {"sigma": sigma, "N_prog": N_prog, "hidden_size": h_bar.shape[1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Model directory containing program representations and index")
    parser.add_argument("--output-dir", type=Path, help="Directory for sigma.json; defaults to --input-dir")
    parser.add_argument("--layer", type=int, help="Explicit zero-indexed layer; otherwise use selected_layer.json")
    parser.add_argument("--limit", type=int, help="Use the first N programs in input-index order (at least 2)")
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
    records, input_count = load_programs(args.input_dir / "program_index.jsonl", args.limit)
    output = (args.output_dir if args.output_dir is not None else args.input_dir) / "sigma.json"
    if output.exists():
        raise FileExistsError(f"Results already exist in {output}; choose another --output-dir")
    h_bar = np.load(args.input_dir / "program_representations" / f"layer_{layer:02d}.npy",
                    mmap_mode="r", allow_pickle=False)
    if h_bar.ndim != 2 or h_bar.shape[0] != input_count or h_bar.shape[1] == 0 or h_bar.dtype != np.float32:
        raise ValueError("Expected float32 representations with one row per indexed program; finish step 01 first")
    print(f"Using {len(records)} programs at layer {layer}; sigma calibration runs on the CPU.")
    result = calibrate_sigma(h_bar, records)
    cell_counts = {(language, misuse): 0 for language in LANGUAGES for misuse in MISUSES}
    for record in records:
        cell_counts[record["language"], record["misuse_id"]] += 1
    result.update({"layer": layer, "layer_source": "explicit" if args.layer is not None else "selected_layer.json",
                   "limit": args.limit, "full_source": args.limit is None and all(n == 2400 for n in cell_counts.values()),
                   "population": "all_programs" if args.limit is None else "first_N_programs",
                   "readout": "code_body_mean", "ddof": 1,
                   "formula": "sqrt(sum_j(||h_bar_j - mu_h||^2) / (N_prog - 1))"})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"sigma = {result['sigma']:.17g}")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
