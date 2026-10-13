"""Fit additive repair responses and predict combinations from eleven reference programs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


LANGUAGES = ("java", "typescript", "csharp", "python")
MISUSES = tuple(f"M{i:02d}" for i in range(1, 11))


def load_combinations(root):
    with (root / "combination_index.jsonl").open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if len(records) != 4096 or len({r["id"] for r in records}) != len(records):
        raise ValueError("Finish step 01 on all 4,096 combination programs first")
    rows = [r["row"] for r in records]
    if any(type(row) is not int for row in rows) or sorted(rows) != list(range(len(records))):
        raise ValueError("Index rows must cover the representation array without gaps")
    groups = {language: [] for language in LANGUAGES}
    expected = {f"{i:010b}" for i in range(1024)}
    for record in records:
        language, bits = record["language"], record["secure_bits"]
        if language not in groups or bits not in expected:
            raise ValueError(f"Unknown language or security configuration in {record['id']}")
        if (record["id"] != f"combination__{language}__{bits}"
                or record["secure_count"] != bits.count("1")):
            raise ValueError(f"Program ID or secure count disagrees with secure_bits: {record['id']}")
        groups[language].append(record)
    for language, members in groups.items():
        if len(members) != 1024 or {r["secure_bits"] for r in members} != expected:
            raise ValueError(f"Expected all 1,024 configurations for {language}")
        # Descending secure masks keep the numerical row order of the paper analysis.
        members.sort(key=lambda r: r["secure_bits"], reverse=True)
    return groups, len(records)


def load_direction(root):
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    if type(selection["layer"]) is not int or selection["layer"] < 0:
        raise ValueError("Direction selection must specify its zero-indexed layer")
    direction = np.load(root / "v_CrypSteer.npy", allow_pickle=False)
    if (direction.ndim != 1 or not direction.size or direction.dtype != np.float32
            or not np.isfinite(direction).all()
            or not np.isclose(np.linalg.norm(direction), 1.0, atol=2e-6, rtol=0)):
        raise ValueError("Expected step 04's saved float32 CrypSteer unit direction")
    # Use the saved sign and length; do not fit or normalize the direction again.
    return direction, selection


def r_squared(observed, predicted):
    if len(observed) < 2:
        return None
    variance = float(np.var(observed))
    if variance == 0.0:
        return None
    return 1.0 - float(np.mean((observed - predicted) ** 2)) / variance


def analyze_language(h_bar, records, v_CrypSteer, limit=None):
    if limit is not None and (type(limit) is not int or not 1 <= limit <= 1013):
        raise ValueError("--limit must be between 1 and 1,013 prediction programs per language")
    if (h_bar.dtype != np.float32 or h_bar.shape != (1024, len(v_CrypSteer))
            or not np.isfinite(h_bar).all() or len(records) != 1024):
        raise ValueError("Expected 1,024 finite float32 program representations per language")
    if len({r["language"] for r in records}) != 1:
        raise ValueError("Analyze one language at a time")
    if {r["secure_bits"] for r in records} != {f"{i:010b}" for i in range(1024)}:
        raise ValueError("Expected each ten-bit security configuration exactly once")
    order = sorted(range(len(records)), key=lambda i: records[i]["secure_bits"], reverse=True)
    records = [records[i] for i in order]
    h_bar = np.ascontiguousarray(h_bar[order], dtype=np.float32)
    z = np.asarray([[int(bit) for bit in r["secure_bits"]] for r in records], dtype=np.float64)
    secure_count = z.sum(axis=1).astype(int)
    base = int(np.flatnonzero(secure_count == 0)[0])
    endpoint = int(np.flatnonzero(secure_count == 10)[0])
    singles = [int(np.flatnonzero((secure_count == 1) & (z[:, g] == 1))[0]) for g in range(10)]
    retained = np.ones(len(records), dtype=bool)
    if limit is not None:
        # Retain all eleven anchors, then the first N other configurations in
        # ascending secure-mask order. The full-secure endpoint only sets the plot origin.
        candidates = sorted(np.flatnonzero(secure_count >= 2), key=lambda i: records[i]["secure_bits"])
        retained = secure_count <= 1
        retained[candidates[:limit]] = True

    # Match the descriptive figures: float32 projections and endpoint centering,
    # followed by a float64 least-squares fit with an intercept and ten secure bits.
    raw = h_bar @ v_CrypSteer
    p_midpoint = (raw - np.float32((raw[endpoint] + raw[base]) / 2)).astype(np.float64)
    design = np.column_stack((np.ones(len(records)), z))[retained]
    beta, _, rank, _ = np.linalg.lstsq(design, p_midpoint[retained], rcond=None)
    if rank != 11:
        raise ValueError("The intercept and ten repair contributions must be identifiable")
    p_fit = design @ beta
    total = float(np.sum((p_midpoint[retained] - p_midpoint[retained].mean()) ** 2))
    fit_r2 = (None if total == 0.0 else
              1.0 - float(np.sum((p_midpoint[retained] - p_fit) ** 2)) / total)

    # Match the prediction analysis: subtract the all-misuse representation in
    # float64 before projecting. Only the eleven reference programs determine p_pred.
    h64 = h_bar.astype(np.float64)
    p_from_misuse = (h64 - h64[base]) @ v_CrypSteer.astype(np.float64)
    single_effects = p_from_misuse[singles]
    p_pred = z @ single_effects
    prediction_rows = retained & (secure_count >= 2)
    prediction_r2 = r_squared(p_from_misuse[prediction_rows], p_pred[prediction_rows])

    selected = np.flatnonzero(retained)
    points = []
    for index, fitted in zip(selected, p_fit):
        record = records[index]
        points.append({
            "id": record["id"], "language": record["language"],
            "secure_bits": record["secure_bits"], "secure_count": int(secure_count[index]),
            "role": "anchor" if secure_count[index] <= 1 else "prediction",
            "p_midpoint": float(p_midpoint[index]), "p_fit": float(fitted),
            "p_from_misuse": float(p_from_misuse[index]), "p_pred": float(p_pred[index]),
        })
    metrics = {
        "language": records[0]["language"], "programs": len(selected), "anchors": 11,
        "prediction_programs": int(prediction_rows.sum()),
        "fit_r2": fit_r2, "prediction_r2": prediction_r2,
    }
    coefficients = {"beta0": float(beta[0])}
    coefficients.update({f"beta_{misuse}": float(beta[g + 1]) for g, misuse in enumerate(MISUSES)})
    coefficients.update({f"single_effect_{misuse}": float(single_effects[g]) for g, misuse in enumerate(MISUSES)})
    return points, metrics, coefficients


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Model directory with combination_representations/ and combination_index.jsonl")
    parser.add_argument("--direction-dir", type=Path,
                        help="Step 04 direction folder; defaults to --input-dir/directions")
    parser.add_argument("--output-dir", type=Path,
                        help="Model directory for combination_analysis/; defaults to --input-dir")
    parser.add_argument("--limit", type=int,
                        help="First N multi-repair configurations per language, plus all eleven anchors")
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= 1013:
        parser.error("--limit must be between 1 and 1,013")
    direction_root = args.direction_dir if args.direction_dir is not None else args.input_dir / "directions"
    v_CrypSteer, selection = load_direction(direction_root)
    groups, count = load_combinations(args.input_dir)
    layer = selection["layer"]
    array = np.load(args.input_dir / "combination_representations" / f"layer_{layer:02d}.npy",
                    mmap_mode="r", allow_pickle=False)
    if array.dtype != np.float32 or array.shape != (count, len(v_CrypSteer)):
        raise ValueError("Layer array must have one float32 row per index record and match the direction width")
    output = (args.output_dir if args.output_dir is not None else args.input_dir) / "combination_analysis"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Results already exist in {output}; choose another --output-dir")
    print(f"Layer {layer}, frozen CrypSteer direction; CPU analysis of four languages.")
    points, results, coefficients = [], [], {}
    for language, records in tqdm(groups.items(), desc="Combination analysis", unit="language"):
        h_bar = np.ascontiguousarray(array[[r["row"] for r in records]], dtype=np.float32)
        rows, metrics, terms = analyze_language(h_bar, records, v_CrypSteer, args.limit)
        points.extend(rows)
        results.append(metrics)
        coefficients[language] = terms
    summary = {
        "layer": layer, "direction": "CrypSteer", "selected_pc": selection["selected_pc"],
        "direction_full_source": selection["full_source"],
        "direction_norm": float(np.linalg.norm(v_CrypSteer.astype(np.float64))),
        "languages": list(groups), "programs": len(points),
        "full_factorial": all(r["programs"] == 1024 for r in results),
        "limit_prediction_programs_per_language": args.limit,
        "bit_order": list(MISUSES), "encoding": {"1": "secure", "0": "misuse"},
        "fit": "OLS with intercept and ten secure-choice indicators; fit and score on all retained programs",
        "prediction": "All-misuse projection plus ten single-repair effects; score only multi-repair programs",
        "fit_projection": "float32 dot product, centered at the midpoint of both endpoints; float64 OLS",
        "prediction_projection": "float64 dot product after subtracting the all-misuse representation",
        "r2_denominator": "Centered variation of each analysis's own evaluation set",
        "undefined_r2": "An empty CSV field when fewer than two values or zero variance",
        "language_coefficients": coefficients,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "projections.csv", points)
    write_csv(output / "by_language.csv", results)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for row in results:
        fit = "undefined" if row["fit_r2"] is None else f"{row['fit_r2']:.6f}"
        prediction = "undefined" if row["prediction_r2"] is None else f"{row['prediction_r2']:.6f}"
        print(f"{row['language']}: fit R2={fit}; prediction R2={prediction} ({row['prediction_programs']} programs)")
    print(f"Saved projections.csv, by_language.csv, and summary.json to {output}")


if __name__ == "__main__":
    main()
