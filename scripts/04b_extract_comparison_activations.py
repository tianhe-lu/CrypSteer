import argparse
import json
from pathlib import Path
import random


LANGUAGE_NAMES = {"java": "Java", "typescript": "TypeScript", "csharp": "C#", "python": "Python"}


def load_scs_pairs(path, limit=None):
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1")
    grouped = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            language, misuse, task = (record[name] for name in ("language", "misuse_id", "task_id"))
            prefix = f"{language}__{misuse}_{task}_"
            if language not in LANGUAGE_NAMES or not record["original_id"].startswith(prefix):
                raise ValueError(f"Unexpected original ID: {record['original_id']}")
            sample = record["original_id"][len(prefix):]
            key = language, misuse, task, sample, record["structure"]
            pair = grouped.setdefault(key, {
                "pair_id": record["pair_id"], "original_id": record["original_id"],
                "language": language, "misuse_id": misuse, "task_id": task,
                "sample_id": sample, "structure": record["structure"],
            })
            label = record["label"]
            if label not in ("secure", "misuse") or f"{label}_code" in pair:
                raise ValueError(f"Duplicate or invalid label for {record['pair_id']}")
            if pair["pair_id"] != record["pair_id"] or not record["code"].strip():
                raise ValueError(f"Invalid paired program: {record['pair_id']}")
            pair[f"{label}_code"] = record["code"].replace("\r\n", "\n").replace("\r", "\n")
    pairs = [grouped[key] for key in sorted(grouped)]
    if not pairs or len(pairs) % 2:
        raise ValueError("SCS requires an even number of complete program pairs")
    if any("secure_code" not in pair or "misuse_code" not in pair for pair in pairs):
        raise ValueError("Every SCS pair must have both secure and misuse programs")
    if len({pair["pair_id"] for pair in pairs}) != len(pairs):
        raise ValueError("SCS pair IDs must be unique")
    options = ["A"] * (len(pairs) // 2) + ["B"] * (len(pairs) // 2)
    random.Random(2026).shuffle(options)
    for pair, option in zip(pairs, options):
        pair["secure_option"] = option
    return pairs[:limit], len(pairs)


def load_surgical_pairs(path, limit=None):
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1")
    with path.open(encoding="utf-8") as handle:
        pairs = [json.loads(line) for line in handle]
    if not pairs or len({p["pair_id"] for p in pairs}) != len(pairs):
        raise ValueError("Surgical instruction pairs must have unique IDs")
    for pair in pairs:
        if pair["language"] not in LANGUAGE_NAMES or any(
                not isinstance(pair.get(f"{label}_input"), str) or not pair[f"{label}_input"].strip()
                for label in ("secure", "misuse")):
            raise ValueError(f"Invalid Surgical instruction pair: {pair['pair_id']}")
    return pairs[:limit], len(pairs)


def make_inputs(method, pair):
    if method == "surgical":
        return pair["secure_input"], pair["misuse_input"]
    if method != "scs":
        raise ValueError(f"Unknown comparison method: {method}")
    language = pair["language"]
    name = LANGUAGE_NAMES[language]
    secure_option = pair["secure_option"]
    misuse_option = "B" if secure_option == "A" else "A"
    code_a = pair["secure_code"] if secure_option == "A" else pair["misuse_code"]
    code_b = pair["misuse_code"] if secure_option == "A" else pair["secure_code"]
    question = (
        f"Hello, I am a researcher on {name} programming code. I will give you "
        f"two {name} code snippets in the following, marked with (A) and (B). "
        "You need to choose one code snippet. Here are the snippets:\n"
        f"(A) ```{language}\n{code_a}\n```\n\n"
        f"(B) ```{language}\n{code_b}\n```\n\n"
        "Which snippet do you choose, (A) or (B)?"
    )
    return question + f"\n({secure_option})", question + f"\n({misuse_option})"


def locate_readout(text, offsets, attention, special):
    last = len(text.rstrip()) - 1
    if last < 0 or not (len(offsets) == len(attention) == len(special)):
        raise ValueError("Empty input or inconsistent tokenizer offsets")
    positions = [i for i, (start, end) in enumerate(offsets)
                 if attention[i] and not special[i] and start <= last < end]
    if not positions:
        raise ValueError("No active text token covers the final non-whitespace character")
    return positions[-1]


def extract_activation(model, tokenizer, text, layer):
    import torch

    inputs = tokenizer([text], return_tensors="pt", return_offsets_mapping=True,
                       return_special_tokens_mask=True, add_special_tokens=True,
                       truncation=False, padding=False)
    offsets = inputs.pop("offset_mapping")[0].tolist()
    special = inputs.pop("special_tokens_mask")[0].tolist()
    token_ids = inputs["input_ids"][0].tolist()
    attention = inputs["attention_mask"][0].tolist() if "attention_mask" in inputs else [1] * len(token_ids)
    position = locate_readout(text, offsets, attention, special)
    device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    captured = []

    def capture(_module, _inputs, output):
        h = output[0] if isinstance(output, (tuple, list)) else output
        captured.append(h[0, position, :].detach().to(device="cpu", dtype=torch.float32).numpy().copy())

    handle = model.model.layers[layer].register_forward_hook(capture)
    try:
        with torch.inference_mode():
            decoder = getattr(model, "base_model", model)
            decoder(**inputs, output_hidden_states=False, use_cache=False)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"Expected one block activation, received {len(captured)}")
    return captured[0], {"input_token_count": len(token_ids), "readout_position": position,
                         "readout_token_id": token_ids[position], "readout_offset": offsets[position]}


def main():
    artifacts = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Model JSON, e.g. config/qwen.json")
    parser.add_argument("--input-dir", type=Path, help="Model folder containing layer_probes/")
    parser.add_argument("--programs", type=Path, default=artifacts / "data/programs.jsonl")
    parser.add_argument("--surgical-pairs", type=Path, default=artifacts / "data/surgical_prompt_pairs.jsonl")
    parser.add_argument("--methods", choices=("scs", "surgical"), nargs="+", default=["scs", "surgical"])
    parser.add_argument("--output-dir", type=Path, default=artifacts / "outputs",
                        help="Parent output folder; results go under MODEL/comparison_activations/")
    parser.add_argument("--layer", type=int, help="Zero-indexed decoder block; otherwise use step 03")
    parser.add_argument("--limit", type=int, help="Extract the first N complete pairs per selected method")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    input_dir = args.input_dir or artifacts / "outputs" / config["model_name"]
    layer = args.layer
    if layer is None:
        selection = input_dir / "layer_probes/selected_layer.json"
        layer = json.loads(selection.read_text(encoding="utf-8"))["selected_layer"]
    if type(layer) is not int or layer < 0:
        parser.error("Layer must be a nonnegative integer")
    methods = list(dict.fromkeys(args.methods))
    output = args.output_dir / config["model_name"] / "comparison_activations"
    for method in methods:
        directory = output / method
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise FileExistsError(f"Results already exist in {directory}; choose another --output-dir")
    pairs, populations = {}, {}
    for method in methods:
        loader, path = (load_scs_pairs, args.programs) if method == "scs" else (load_surgical_pairs, args.surgical_pairs)
        pairs[method], populations[method] = loader(path, args.limit)

    import numpy as np
    import torch
    import transformers
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_path"], use_fast=True, trust_remote_code=config["trust_remote_code"],
        local_files_only=args.local_files_only)
    if not tokenizer.is_fast:
        raise ValueError("Extraction requires a fast tokenizer with character offsets")
    tokenizer.padding_side = "left"
    options = {"dtype": getattr(torch, config["dtype"]), "trust_remote_code": config["trust_remote_code"],
               "local_files_only": args.local_files_only}
    if device == "cuda":
        options["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(config["model_path"], **options)
    if device == "cpu":
        model.to("cpu")
    model.eval()
    if layer >= model.config.num_hidden_layers:
        raise ValueError("Selected layer is outside the loaded model")
    hidden_size = model.config.hidden_size
    runtime = {"torch": torch.__version__, "transformers": transformers.__version__,
               "numpy": np.__version__, "device": device, "cuda": torch.version.cuda,
               "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
               if device == "cuda" else []}
    for method in methods:
        directory = output / method
        directory.mkdir(parents=True, exist_ok=True)
        count = len(pairs[method])
        arrays = {label: np.lib.format.open_memmap(directory / f"h_{label}.npy", mode="w+",
                                                 dtype=np.float32, shape=(count, hidden_size))
                  for label in ("secure", "misuse")}
        settings = {"model_config": config, "method": method, "layer": layer,
                    "layer_source": "explicit" if args.layer is not None else "selected_layer.json",
                    "hidden_size": hidden_size,
                    "pairs": count, "population_pairs": populations[method], "limit_pairs": args.limit,
                    "full_source": count == populations[method] == (48000 if method == "scs" else 144),
                    "serialization": "plain_text", "readout": "last_non_whitespace_character_token",
                    "add_special_tokens": True, "truncation": False, "batch_size": 1,
                    "scs_option_seed": 2026 if method == "scs" else None, "runtime": runtime}
        (directory / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        with (directory / "index.jsonl").open("w", encoding="utf-8") as handle:
            for row, pair in enumerate(tqdm(pairs[method], desc=f"{config['model_name']}: {method}", unit="pair")):
                texts = make_inputs(method, pair)
                index = {key: value for key, value in pair.items()
                         if key not in ("secure_code", "misuse_code", "secure_input", "misuse_input")}
                index["row"] = row
                for label, text in zip(("secure", "misuse"), texts):
                    h, readout = extract_activation(model, tokenizer, text, layer)
                    if h.shape != (hidden_size,) or not np.isfinite(h).all():
                        raise ValueError(f"Invalid activation: {pair['pair_id']}/{label}")
                    arrays[label][row] = h
                    index[f"{label}_readout"] = readout
                for array in arrays.values():
                    array.flush()
                handle.write(json.dumps(index, ensure_ascii=False) + "\n")
                handle.flush()
        print(f"Saved {count} pairs at layer {layer}: {directory}")


if __name__ == "__main__":
    main()
