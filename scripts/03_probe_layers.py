import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch.nn import functional
from tqdm import tqdm


LANGUAGES = ("java", "typescript", "csharp", "python")
MISUSES = tuple(f"M{i:02d}" for i in range(1, 11))
STRUCTURES = ("variable_inline", "callsite_inline", "variable_helper", "callsite_helper")


def load_population(index_path, limit):
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1")
    with index_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    if not records:
        raise ValueError("The program index is empty")
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("Duplicate program IDs")
    rows = [record["row"] for record in records]
    if any(type(row) is not int for row in rows) or sorted(rows) != list(range(len(records))):
        raise ValueError("Program index rows must cover 0 through the number of programs minus one")

    originals = {}
    for record in records:
        if record["language"] not in LANGUAGES or record["misuse_id"] not in MISUSES:
            raise ValueError(f"Unexpected language or misuse category in {record['id']}")
        originals.setdefault(record["original_id"], []).append(record)
    cells = {(language, misuse): [] for language in LANGUAGES for misuse in MISUSES}
    expected = {(structure, label) for structure in STRUCTURES for label in ("secure", "misuse")}
    for original_id, variants in originals.items():
        metadata = {(r["language"], r["misuse_id"], r["task_id"]) for r in variants}
        if len(metadata) != 1 or len(variants) != 8 or {
            (r["structure"], r["label"]) for r in variants
        } != expected:
            raise ValueError(f"{original_id} needs four complete, consistent secure/misuse pairs")
        language, misuse, _ = next(iter(metadata))
        cells[language, misuse].append(original_id)
    selected = set()
    for cell, ids in cells.items():
        if not ids or (limit is not None and len(ids) < limit):
            raise ValueError(f"{cell} has {len(ids)} complete originals; all 40 cells are required")
        selected.update(ids[:limit])

    population = sorted(
        (record for record in records if record["original_id"] in selected),
        key=lambda r: (MISUSES.index(r["misuse_id"]), LANGUAGES.index(r["language"]),
                       r["task_id"], r["original_id"], r["id"]),
    )
    return population, len(records)


def sample_weights(labels, groups, languages, misuses):
    counts = Counter(zip(groups, labels))
    weights = np.asarray([1.0 / counts[group, label] for group, label in zip(groups, labels)])
    for language, misuse in sorted(set(zip(languages.tolist(), misuses.tolist()))):
        cell = (languages == language) & (misuses == misuse)
        weights[cell] /= float(weights[cell].sum())
    return weights * (len(weights) / float(weights.sum()))


def make_folds(population):
    labels = np.array([r["label"] == "secure" for r in population], dtype=np.int8)
    groups = np.array([r["original_id"] for r in population])
    languages = np.array([r["language"] for r in population])
    misuses = np.array([r["misuse_id"] for r in population])
    folds = []
    for protocol, values, holdouts in (
        ("leave_one_misuse_out", misuses, MISUSES),
        ("leave_one_language_out", languages, LANGUAGES),
    ):
        for heldout in holdouts:
            test = values == heldout
            train = ~test
            train_weights = np.zeros((len(labels), 1), dtype=np.float64)
            train_weights[train, 0] = sample_weights(
                labels[train], groups[train], languages[train], misuses[train])
            test_weights = sample_weights(labels[test], groups[test], languages[test], misuses[test])
            folds.append({"protocol": protocol, "heldout": heldout, "test": test,
                          "train_weights": train_weights, "test_weights": test_weights})
    return labels, folds


def fit_probe(h_bar, labels, weights, config):
    w = torch.zeros((1, h_bar.shape[1]), dtype=h_bar.dtype, device=h_bar.device, requires_grad=True)
    b = torch.zeros(1, dtype=h_bar.dtype, device=h_bar.device, requires_grad=True)
    C = torch.as_tensor((config["regularization_c"],), dtype=h_bar.dtype, device=h_bar.device)
    weight_sum = weights.sum(dim=0)
    optimizer = torch.optim.LBFGS(
        (w, b), lr=1.0, max_iter=config["max_iterations"],
        tolerance_grad=config["tolerance"], tolerance_change=torch.finfo(h_bar.dtype).eps,
        history_size=20, line_search_fn="strong_wolfe",
    )

    def objective():
        logits = h_bar @ w.T + b
        losses = functional.binary_cross_entropy_with_logits(
            logits, labels[:, None].expand_as(logits), reduction="none")
        return ((losses * weights).sum(dim=0) / weight_sum
                + 0.5 * w.square().sum(dim=1) / (C * weight_sum)).sum()

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        return loss

    optimizer.step(closure)
    optimizer.zero_grad(set_to_none=True)
    loss = objective()
    loss.backward()
    gradient = torch.maximum(w.grad.detach().abs().amax(dim=1), b.grad.detach().abs()).item()
    state = optimizer.state[w]
    iterations = int(state["n_iter"])
    with torch.no_grad():
        probabilities = torch.sigmoid(h_bar @ w.T + b).cpu().numpy()[:, 0]
    diagnostics = {
        "iterations": iterations, "function_evaluations": int(state["func_evals"]),
        "objective": loss.item(), "gradient_inf_norm": gradient,
        "converged": bool(np.isfinite(loss.item()) and gradient <= config["tolerance"]
                          and iterations < config["max_iterations"]),
    }
    return probabilities, diagnostics


def summarize_layer(layer, results):
    misuse_auc = float(np.mean([r["heldout_auroc"] for r in results
                               if r["protocol"] == "leave_one_misuse_out"]))
    language_auc = float(np.mean([r["heldout_auroc"] for r in results
                                 if r["protocol"] == "leave_one_language_out"]))
    return {"layer": layer, "misuse_auroc": misuse_auc, "language_auroc": language_auc,
            "joint_auroc": 0.5 * misuse_auc + 0.5 * language_auc,
            "converged_folds": sum(r["converged"] for r in results), "folds": len(results)}


def select_layer(summaries):
    return max(summaries, key=lambda r: (r["joint_auroc"], -r["layer"]))


def main():
    parser = argparse.ArgumentParser(description="Select a layer using category and language held-out linear probes.")
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Model directory containing program_index.jsonl and program_representations/")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config/probe.json")
    parser.add_argument("--output-dir", type=Path, help="Model directory for layer_probes/; defaults to --input-dir")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--limit", type=int, help="Use the first N complete originals from each language-misuse cell")
    parser.add_argument("--layers", type=int, nargs="+", help="Test only these zero-indexed layers")
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not (np.isfinite(config["regularization_c"]) and config["regularization_c"] > 0
            and np.isfinite(config["tolerance"]) and config["tolerance"] > 0
            and type(config["max_iterations"]) is int and config["max_iterations"] > 0):
        raise ValueError("C, tolerance, and the integer iteration limit must be positive")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu for a small test")

    population, input_count = load_population(args.input_dir / "program_index.jsonl", args.limit)
    paths = sorted((args.input_dir / "program_representations").glob("layer_*.npy"))
    if not paths or [int(p.stem.split("_")[-1]) for p in paths] != list(range(len(paths))):
        raise ValueError("Expected consecutive layer files starting with layer_00.npy")
    layers = list(range(len(paths))) if args.layers is None else sorted(set(args.layers))
    if not layers or min(layers) < 0 or max(layers) >= len(paths):
        raise ValueError("Requested layer is not present in the representations")
    hidden_size = None
    for path in paths:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 2 or array.shape[0] != input_count:
            raise ValueError(f"{path.name}: rows do not match program_index.jsonl; finish extraction first")
        if array.dtype != np.dtype("float32") or array.shape[1] == 0:
            raise ValueError(f"{path.name}: expected nonempty float32 representations")
        if hidden_size is not None and array.shape[1] != hidden_size:
            raise ValueError(f"Hidden dimension differs in {path.name}")
        hidden_size = array.shape[1]
        del array

    output = (args.output_dir if args.output_dir is not None else args.input_dir) / "layer_probes"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Results already exist in {output}; choose another --output-dir")
    output.mkdir(parents=True, exist_ok=True)
    rows = np.array([r["row"] for r in population])
    labels, folds = make_folds(population)
    targets = torch.as_tensor(labels, dtype=torch.float64, device=device)
    fold_fields = ("layer", "protocol", "heldout", "train_programs", "test_programs", "heldout_auroc",
                   "iterations", "function_evaluations", "objective", "gradient_inf_norm", "converged", "seconds")
    summary_fields = ("layer", "misuse_auroc", "language_auroc", "joint_auroc", "converged_folds", "folds")
    summaries = []
    start = time.perf_counter()
    print(f"Using {len(population)} programs across {len(layers)} layers; 14 folds per layer on {device}.")
    with (output / "fold_scores.csv").open("w", newline="", encoding="utf-8") as fold_handle, \
            (output / "layer_scores.csv").open("w", newline="", encoding="utf-8") as summary_handle, \
            tqdm(total=len(layers) * len(folds), desc="Layer probes", unit="fold") as progress:
        fold_writer = csv.DictWriter(fold_handle, fieldnames=fold_fields)
        summary_writer = csv.DictWriter(summary_handle, fieldnames=summary_fields)
        fold_writer.writeheader()
        summary_writer.writeheader()
        for layer in layers:
            array = np.load(paths[layer], mmap_mode="r", allow_pickle=False)
            matrix = np.ascontiguousarray(array[rows], dtype=np.float32)
            if not np.isfinite(matrix).all():
                raise ValueError(f"Layer {layer} contains non-finite program representations")
            h_bar = torch.as_tensor(matrix, dtype=torch.float64, device=device)
            del array, matrix
            results = []
            for fold in folds:
                progress.set_postfix_str(f"layer {layer}, {fold['heldout']}")
                fold_start = time.perf_counter()
                weights = torch.as_tensor(fold["train_weights"], dtype=torch.float64, device=device)
                probabilities, diagnostics = fit_probe(h_bar, targets, weights, config)
                test = fold["test"]
                result = {
                    "layer": layer, "protocol": fold["protocol"], "heldout": fold["heldout"],
                    "train_programs": int((~test).sum()), "test_programs": int(test.sum()),
                    "heldout_auroc": float(roc_auc_score(labels[test], probabilities[test],
                                                         sample_weight=fold["test_weights"])),
                    **diagnostics, "seconds": time.perf_counter() - fold_start,
                }
                results.append(result)
                fold_writer.writerow(result)
                fold_handle.flush()
                del weights
                progress.update(1)
            summaries.append(summarize_layer(layer, results))
            summary_writer.writerow(summaries[-1])
            summary_handle.flush()
            del h_bar

    selected = select_layer(summaries)
    full_corpus = (len(population) == 96000 and args.limit is None and args.layers is None)
    selection = {
        "selected_layer": selected["layer"], "joint_auroc": selected["joint_auroc"],
        "misuse_auroc": selected["misuse_auroc"], "language_auroc": selected["language_auroc"],
        "full_corpus": full_corpus, "programs": len(population), "originals": len(population) // 8,
        "limit_per_cell": args.limit, "layers": layers, "device": device, "dtype": "float64",
        "config": config, "nonconverged_fits": sum(r["folds"] - r["converged_folds"] for r in summaries),
        "elapsed_seconds": time.perf_counter() - start,
    }
    if full_corpus:
        name = "selected_layer.json" if device == "cuda" else "selected_layer_cpu.json"
    else:
        name = "selected_layer_subset.json"
    with (output / name).open("w", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Selected layer {selected['layer']}: joint AUROC {selected['joint_auroc']:.6f}")
    print(f"Saved {output / name}; nonconverged fits: {selection['nonconverged_fits']}")


if __name__ == "__main__":
    main()
