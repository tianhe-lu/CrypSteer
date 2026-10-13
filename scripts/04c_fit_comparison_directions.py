import argparse
from itertools import islice
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


METHODS = ("scs", "surgical")


def load_activations(directory, method, limit=None):
    if method not in METHODS or (limit is not None and limit < 1):
        raise ValueError("Select scs or surgical, with a positive --limit if specified")
    settings = json.loads((directory / "settings.json").read_text(encoding="utf-8"))
    if settings["method"] != method:
        raise ValueError(f"Method differs from the saved activations in {directory}")
    if type(settings["layer"]) is not int or settings["layer"] < 0:
        raise ValueError("Expected a zero-indexed decoder layer")
    if (settings["readout"] != "last_non_whitespace_character_token"
            or settings["serialization"] != "plain_text"
            or settings["add_special_tokens"] is not True or settings["truncation"] is not False):
        raise ValueError("Expected the token readout and input settings from step 04b")
    with (directory / "index.jsonl").open(encoding="utf-8") as handle:
        index = [json.loads(line) for line in islice(handle, limit)]
    planned = settings["pairs"]
    if type(planned) is not int or planned < 1 or len(index) > planned:
        raise ValueError("Invalid pair count in the saved activations")
    if [row["row"] for row in index] != list(range(len(index))):
        raise ValueError("Activation index rows must be consecutive and start at zero")
    if len({row["pair_id"] for row in index}) != len(index):
        raise ValueError("Activation index contains duplicate pair IDs")
    if limit is None and len(index) != planned:
        raise ValueError(f"Step 04b is incomplete: {len(index)}/{planned} pairs in {directory}")
    count = planned if limit is None else limit
    if count > len(index):
        raise ValueError(f"Requested {count} pairs, but only {len(index)} are complete")
    arrays = [np.load(directory / f"h_{label}.npy", mmap_mode="r", allow_pickle=False)
              for label in ("secure", "misuse")]
    expected = (planned, settings["hidden_size"])
    if expected[1] < 1 or any(a.shape != expected or a.dtype != np.float32 for a in arrays):
        raise ValueError("Expected float32 endpoint arrays matching settings and index")
    return arrays[0][:count], arrays[1][:count], settings, index[:count]


def fit_direction(h_secure, h_misuse, method, show_progress=True):
    if method not in METHODS:
        raise ValueError(f"Unknown comparison method: {method}")
    if (h_secure.ndim != 2 or h_secure.shape != h_misuse.shape or not all(h_secure.shape)
            or h_secure.dtype != np.float32 or h_misuse.dtype != np.float32):
        raise ValueError("Expected two matching nonempty float32 endpoint matrices")
    count, hidden_size = h_secure.shape
    if method == "scs":
        secure_sum = np.zeros(hidden_size, dtype=np.float64)
        misuse_sum = np.zeros(hidden_size, dtype=np.float64)
    else:
        differences = np.empty((count, hidden_size), dtype=np.float32)
    for row in tqdm(range(count), desc=f"Fit {method}", unit="pair", disable=not show_progress):
        secure, misuse = h_secure[row], h_misuse[row]
        if not np.isfinite(secure).all() or not np.isfinite(misuse).all():
            raise ValueError(f"Non-finite activation at pair row {row}")
        if method == "scs":
            secure_sum += secure
            misuse_sum += misuse
        else:
            differences[row] = secure - misuse
    if method == "scs":
        mean_difference = (secure_sum / count - misuse_sum / count).astype(np.float32)
    else:
        mean_difference = differences.mean(axis=0, dtype=np.float32)
    tensor = torch.from_numpy(mean_difference)
    norm = tensor.norm()
    if not torch.isfinite(norm) or float(norm) <= 0:
        raise ValueError("The shared mean cannot be normalized into a direction")
    unit = (tensor.clone() / norm).numpy()
    return mean_difference, unit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Model folder containing comparison_activations/ from step 04b")
    parser.add_argument("--output-dir", type=Path,
                        help="Model folder for comparison_directions/; defaults to --input-dir")
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    parser.add_argument("--limit", type=int, help="Fit on the first N complete pairs per method")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    methods = list(dict.fromkeys(args.methods))
    output = (args.output_dir or args.input_dir) / "comparison_directions"
    for method in methods:
        directory = output / method
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise FileExistsError(f"Results already exist in {directory}; choose another --output-dir")
    inputs = {method: load_activations(args.input_dir / "comparison_activations" / method,
                                       method, args.limit) for method in methods}
    first = inputs[methods[0]][2]
    for _, _, settings, _ in inputs.values():
        if any(settings[key] != first[key] for key in ("model_config", "layer", "hidden_size")):
            raise ValueError("The selected methods must use the same model and decoder layer")
    for method in methods:
        secure, misuse, source, index = inputs[method]
        mean_difference, unit = fit_direction(secure, misuse, method)
        directory = output / method
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / f"v_{method}.npy", unit, allow_pickle=False)
        settings = {
            "model_config": source["model_config"], "method": method, "layer": source["layer"],
            "hidden_size": len(unit), "pairs": len(index), "source_pairs": source["pairs"],
            "population_pairs": source["population_pairs"], "limit_pairs": args.limit,
            "full_source": bool(source["full_source"] and len(index) == source["population_pairs"]
                                == (48000 if method == "scs" else 144)),
            "aggregation": "difference_of_endpoint_means" if method == "scs" else "mean_pair_difference",
            "aggregation_dtype": "float64" if method == "scs" else "float32",
            "raw_mean_norm": float(np.linalg.norm(mean_difference.astype(np.float64))),
            "unit_normalized": True, "normalization_dtype": "float32",
        }
        (directory / "settings.json").write_text(
            json.dumps(settings, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Saved {method} unit direction from {len(index)} pairs: {directory}")


if __name__ == "__main__":
    main()
