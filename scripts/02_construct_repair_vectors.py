import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


STRUCTURES = ("variable_inline", "callsite_inline", "variable_helper", "callsite_helper")


def load_pairs(index_path, limit):
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1")
    with index_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    if not records:
        raise ValueError("The program index is empty")
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("Duplicate program IDs in the input index")
    rows = [record["row"] for record in records]
    if any(type(row) is not int for row in rows) or sorted(rows) != list(range(len(records))):
        raise ValueError("Input row indices must cover 0 through the number of programs minus one")

    groups, pair_owners = {}, {}
    for record in records:
        original_id, structure, label = record["original_id"], record["structure"], record["label"]
        if structure not in STRUCTURES or label not in {"secure", "misuse"}:
            raise ValueError(f"Unexpected structure or label in {record['id']}")
        owner = (original_id, structure)
        if pair_owners.setdefault(record["pair_id"], owner) != owner:
            raise ValueError(f"Pair ID belongs to different originals or structures: {record['pair_id']}")
        by_label = groups.setdefault(original_id, {}).setdefault(structure, {})
        if label in by_label:
            raise ValueError(f"Duplicate {label} program for {original_id}, {structure}")
        by_label[label] = record

    complete, incomplete = [], []
    for original_id, by_structure in groups.items():
        metadata = {
            (record["language"], record["misuse_id"], record["task_id"])
            for by_label in by_structure.values() for record in by_label.values()
        }
        if len(metadata) != 1:
            raise ValueError(f"Inconsistent language, misuse, or task in {original_id}")
        for by_label in by_structure.values():
            if len(by_label) == 2 and by_label["secure"]["pair_id"] != by_label["misuse"]["pair_id"]:
                raise ValueError(f"Secure and misuse pair IDs differ in {original_id}")
        if len(by_structure) == 4 and all(len(by_label) == 2 for by_label in by_structure.values()):
            complete.append(original_id)
        else:
            incomplete.append(original_id)

    if limit is None and incomplete:
        raise ValueError(f"Incomplete original: {incomplete[0]}; each original needs four secure/misuse pairs")
    if limit is not None and len(complete) < limit:
        raise ValueError(f"Requested {limit} complete originals, but found {len(complete)}")
    selected = complete[:limit]
    if not selected:
        raise ValueError("No complete originals found")

    delta_index, d_index = [], []
    for original_id in selected:
        pair_rows = []
        for structure in STRUCTURES:
            secure = groups[original_id][structure]["secure"]
            misuse = groups[original_id][structure]["misuse"]
            metadata = {key: secure[key] for key in ("original_id", "language", "misuse_id", "task_id")}
            pair_rows.append(len(delta_index))
            delta_index.append({
                "row": len(delta_index), "pair_id": secure["pair_id"], **metadata,
                "structure": structure,
                "secure_program_id": secure["id"], "misuse_program_id": misuse["id"],
                "secure_row": secure["row"], "misuse_row": misuse["row"],
            })
        d_index.append({"row": len(d_index), **metadata, "pair_rows": pair_rows})
    return delta_index, d_index, len(records)


def repair_vectors(h_bar, secure_rows, misuse_rows):
    secure = np.asarray(h_bar[secure_rows], dtype=np.float32)
    misuse = np.asarray(h_bar[misuse_rows], dtype=np.float32)
    if not np.isfinite(secure).all() or not np.isfinite(misuse).all():
        raise ValueError("Program representations contain non-finite values")
    delta = secure - misuse
    d = delta.mean(axis=0)
    return delta, d


def write_index(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Model directory containing program_index.jsonl and program_representations/")
    parser.add_argument("--output-dir", type=Path,
                        help="Model directory for repair_vectors/; defaults to --input-dir")
    parser.add_argument("--limit", type=int,
                        help="Use the first N complete originals in input-index order")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    delta_index, d_index, program_count = load_pairs(args.input_dir / "program_index.jsonl", args.limit)
    layers = sorted((args.input_dir / "program_representations").glob("layer_*.npy"))
    if not layers or [int(path.stem.split("_")[-1]) for path in layers] != list(range(len(layers))):
        raise ValueError("Expected consecutive layer files starting with layer_00.npy")

    hidden_size = None
    for path in layers:
        h_bar = np.load(path, mmap_mode="r", allow_pickle=False)
        if h_bar.ndim != 2 or h_bar.shape[0] != program_count:
            raise ValueError(f"{path.name}: array rows do not match the program index; finish extraction first")
        if h_bar.dtype != np.dtype("float32") or h_bar.shape[1] == 0:
            raise ValueError(f"{path.name}: expected nonempty float32 representations")
        if hidden_size is None:
            hidden_size = h_bar.shape[1]
        elif h_bar.shape[1] != hidden_size:
            raise ValueError(f"Hidden dimension differs in {path.name}")
        del h_bar

    output = (args.output_dir if args.output_dir is not None else args.input_dir) / "repair_vectors"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Results already exist in {output}; choose another --output-dir")
    (output / "delta").mkdir(parents=True, exist_ok=True)
    (output / "d").mkdir(parents=True, exist_ok=True)

    secure_rows = np.array([row["secure_row"] for row in delta_index]).reshape(-1, 4)
    misuse_rows = np.array([row["misuse_row"] for row in delta_index]).reshape(-1, 4)
    print(f"Using {len(d_index)} complete originals ({len(delta_index)} pairs) across {len(layers)} layers.")
    with tqdm(total=len(layers) * len(d_index), desc="Construct delta and d", unit="original-layer") as progress:
        for path in layers:
            progress.set_postfix_str(path.stem)
            h_bar = np.load(path, mmap_mode="r", allow_pickle=False)
            delta_output = np.lib.format.open_memmap(
                output / "delta" / path.name, mode="w+", dtype=np.float32,
                shape=(len(delta_index), hidden_size),
            )
            d_output = np.lib.format.open_memmap(
                output / "d" / path.name, mode="w+", dtype=np.float32,
                shape=(len(d_index), hidden_size),
            )
            for i in range(len(d_index)):
                delta, d = repair_vectors(h_bar, secure_rows[i], misuse_rows[i])
                delta_output[4 * i:4 * i + 4] = delta
                d_output[i] = d
                progress.update(1)
            delta_output.flush()
            d_output.flush()
            del h_bar, delta_output, d_output

    write_index(output / "delta_index.jsonl", delta_index)
    write_index(output / "d_index.jsonl", d_index)
    print(f"Saved {len(layers)} layers: delta ({len(delta_index)}, {hidden_size}), d ({len(d_index)}, {hidden_size}).")
    print(f"Repair vectors: {output}")


if __name__ == "__main__":
    main()
