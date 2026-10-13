import argparse
import json
import math
from pathlib import Path
import re
import time


DIRECTION_FILES = {"cell": "v_cell.npy", "pooled": "v_pooled.npy",
                   "normalized": "v_norm.npy", "crypsteer": "v_CrypSteer.npy"}
EXCLUDED_METHODS = {f"crypsteer_exclude_{source}": f"exclude_{source}"
                    for source in ("language", "misuse", "both")}
COMPARISON_METHODS = ("scs", "surgical")
FENCE_LABELS = {"java": ("java",), "typescript": ("typescript", "ts"),
                "csharp": ("csharp", "cs", "c#", "dotnet"), "python": ("python", "py")}


def load_prompts(path):
    with path.open(encoding="utf-8") as handle:
        prompts = [json.loads(line) for line in handle]
    ids = [p["prompt_id"] for p in prompts]
    if not prompts or len(ids) != len(set(ids)):
        raise ValueError("Expected nonempty prompts with unique prompt_id values")
    for prompt in prompts:
        if prompt["language"] not in FENCE_LABELS or not prompt["generation_prompt"].strip():
            raise ValueError(f"Invalid generation prompt: {prompt['prompt_id']}")
    return prompts


def extract_program_source(raw_text, language):
    matching, generic = [], []
    for match in re.finditer(r"```(?P<label>[^\r\n`]*)[ \t]*\r?\n?(?P<body>.*?)```",
                             raw_text, flags=re.DOTALL):
        label, body = match.group("label").strip().lower(), match.group("body")
        generic.append(body)
        if label in FENCE_LABELS[language]:
            matching.append(body)
    source = matching[-1] if matching else generic[-1] if generic else raw_text
    if language == "java":
        source = re.sub(r"^\s*package\s+[\w.]+\s*;\s*", "", source, flags=re.MULTILINE)
        if not matching and not generic:
            match = re.search(r"\b(?:public\s+)?class\s+[A-Za-z_]\w*\b", source)
            if match is not None:
                source = source[source.rfind("\n", 0, match.start()) + 1:]
    return source.strip() + "\n"


def make_steering_hook(unit, coefficient):
    cache = {}

    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, (tuple, list)) else output
        key = h.device, h.dtype
        if key not in cache:
            cache[key] = unit.to(device=h.device, dtype=h.dtype) * coefficient
        updated = h.clone()
        updated[:, -1, :] = updated[:, -1, :] + cache[key]
        if isinstance(output, tuple):
            return (updated, *output[1:])
        if isinstance(output, list):
            return [updated, *output[1:]]
        return updated

    return hook


def generation_kwargs(generation, pad_token_id):
    return {"max_new_tokens": generation["max_new_tokens"],
            "temperature": generation["temperature"], "top_p": generation["top_p"],
            "do_sample": generation["do_sample"], "pad_token_id": pad_token_id}


def effective_generation_config(defaults, generation, pad_token_id):
    effective = {**defaults, **generation_kwargs(generation, pad_token_id)}
    return {key: value for key, value in effective.items()
            if value is not None and not key.startswith("_") and key != "transformers_version"}


def generate_one(model, tokenizer, prompt, unit, layer, coefficient, seed, generation):
    import torch
    from transformers import set_seed

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    encoded = tokenizer([rendered], return_tensors="pt")
    input_device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(input_device) for key, value in encoded.items()}
    input_length = inputs["input_ids"].shape[1]
    set_seed(seed)
    handle = None
    if unit is not None:
        handle = model.model.layers[layer].register_forward_hook(make_steering_hook(unit, coefficient))
    try:
        with torch.inference_mode():
            generated = model.generate(
                **inputs, **generation_kwargs(generation, tokenizer.eos_token_id))
    finally:
        if handle is not None:
            handle.remove()
    generated_ids = generated[:, input_length:]
    raw = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return raw, input_length, generated_ids.shape[1]


def method_and_scope(method):
    return ("crypsteer", EXCLUDED_METHODS[method]) if method in EXCLUDED_METHODS else (method, "full")


def load_unit_array(path, shape):
    import numpy as np
    import torch

    values = np.load(path, allow_pickle=False)
    if values.dtype != np.float32 or values.shape != shape or not np.isfinite(values).all():
        raise ValueError(f"Invalid float32 direction array: {path}; expected shape {shape}")
    if not np.allclose(np.linalg.norm(values, axis=-1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError(f"Expected saved unit directions: {path}")
    return torch.from_numpy(values)


def load_target_rows(path, scope):
    with path.open(encoding="utf-8") as handle:
        index = [json.loads(line) for line in handle]
    cells = [(language, f"M{g:02d}") for language in FENCE_LABELS for g in range(1, 11)]
    targets = {(language if scope != "exclude_misuse" else None,
                misuse if scope != "exclude_language" else None) for language, misuse in cells}
    rows = {}
    for item in index:
        key = ((item["language"], item["misuse_id"]) if scope == "cell" else
               (item["target_language"], item["target_misuse"]))
        if key in rows or type(item["row"]) is not int:
            raise ValueError(f"Duplicate target or invalid direction row in {path}")
        rows[key] = item["row"]
    if set(rows) != targets or sorted(rows.values()) != list(range(len(targets))):
        raise ValueError(f"Incomplete or incorrect {scope} direction index: {path}")
    return {(language, misuse): rows[language if scope != "exclude_misuse" else None,
                                     misuse if scope != "exclude_language" else None]
            for language, misuse in cells}, index


def load_directions(input_dir, methods, model_config=None):
    steered = [method for method in methods if method != "baseline"]
    if not steered:
        return {}, {}, None, None, {}
    scale = json.loads((input_dir / "sigma.json").read_text(encoding="utf-8"))
    layer, sigma, hidden_size = scale["layer"], scale["sigma"], scale["hidden_size"]
    if type(layer) is not int or layer < 0:
        raise ValueError("Sigma must specify a zero-indexed decoder layer")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Sigma must be finite and positive")
    scale_details = {"sigma_programs": scale["N_prog"], "sigma_full_source": scale["full_source"]}
    vectors, row_maps, details = {}, {}, {}
    for method in steered:
        if method in DIRECTION_FILES:
            root = input_dir / "directions"
            source = json.loads((root / "selection.json").read_text(encoding="utf-8"))
            path = root / DIRECTION_FILES[method]
            shape = (40, hidden_size) if method == "cell" else (hidden_size,)
            if method == "cell":
                row_maps[method], _ = load_target_rows(root / "cell_index.jsonl", "cell")
            
            details[method] = {"selected_pc": source["selected_pc"], "selected_sign": source["selected_sign"],
                               "direction_originals": source["originals"],
                               "direction_full_source": source["full_source"], **scale_details}
        elif method in EXCLUDED_METHODS:
            scope = EXCLUDED_METHODS[method]
            root = input_dir / "source_excluded_directions" / scope
            source = json.loads((root / "selection.json").read_text(encoding="utf-8"))
            row_maps[method], index = load_target_rows(root / "index.jsonl", scope)
            if source["source_scope"] != scope or source["directions"] != len(index):
                raise ValueError(f"Source-exclusion settings do not match the direction index: {root}")
            path, shape = root / "v_CrypSteer.npy", (len(index), hidden_size)
            details[method] = {"source_scope": scope, "source_originals": source["originals"],
                               "full_retained_source": source["full_retained_source"],
                               "directions": [{key: row[key] for key in (
                                   "row", "target_language", "target_misuse", "originals", "cells",
                                   "selected_pc", "selected_sign")} for row in index], **scale_details}
        elif method in COMPARISON_METHODS:
            root = input_dir / "comparison_directions" / method
            source = json.loads((root / "settings.json").read_text(encoding="utf-8"))
            if source["method"] != method or not source["unit_normalized"]:
                raise ValueError(f"Expected a fitted {method} unit direction in {root}")
            if model_config is not None and source["model_config"] != model_config:
                raise ValueError(f"Comparison direction and generation model configurations differ: {root}")
            path, shape = root / f"v_{method}.npy", (hidden_size,)
            details[method] = {"source_scope": "full", "direction_pairs": source["pairs"],
                               "direction_full_source": source["full_source"],
                               "aggregation": source["aggregation"],
                               "aggregation_dtype": source["aggregation_dtype"], **scale_details}
        else:
            raise ValueError(f"Unknown steering method: {method}")
        if source["layer"] != layer or source["hidden_size"] != hidden_size:
            raise ValueError(f"Directions and sigma must use the same layer and hidden size: {root}")
        vectors[method] = load_unit_array(path, shape)
    return vectors, row_maps, layer, sigma, details


def direction_for_prompt(method, prompt, vectors, row_maps):
    if method == "baseline":
        return None
    if method in row_maps:
        cell = prompt["language"], prompt["misuse_id"]
        if cell not in row_maps[method]:
            raise ValueError(f"No {method} direction for target cell {cell}")
        return vectors[method][row_maps[method][cell]]
    return vectors[method]


def read_existing(path, settings):
    existing = set()
    if not path.exists():
        return existing
    prompts = {p["prompt_id"]: p for p in settings["prompts"]}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            key = record["method"], record["prompt_id"], record["sample_index"]
            method, prompt_id, index = key
            if key in existing or method != settings["method"] or prompt_id not in prompts:
                raise ValueError("Saved outputs contain duplicate or unexpected identities")
            prompt = prompts[prompt_id]
            baseline = method == "baseline"
            expected = {"model": settings["model_config"]["model_name"],
                        "seed": settings["generation"]["seed_start"] + index,
                        "alpha": 0.0 if baseline else settings["alpha"],
                        "layer": None if baseline else settings["layer"],
                        "sigma": None if baseline else settings["sigma"],
                        "effective_coefficient": 0.0 if baseline else settings["alpha"] * settings["sigma"],
                        **{name: prompt[name] for name in ("language", "misuse_id", "task_id")}}
            if not 0 <= index < settings["generation"]["samples_per_prompt"] or any(
                    record.get(name) != value for name, value in expected.items()):
                raise ValueError(f"Saved output settings differ for {key}")
            scope = settings["direction_selection"].get("source_scope", "full")
            if record.get("direction_scope", "full") != scope:
                raise ValueError(f"Saved output direction scope differs for {key}")
            if not isinstance(record.get("raw_response"), str) or not isinstance(record.get("code"), str):
                raise ValueError(f"Missing generated text for {key}")
            existing.add(key)
    return existing


def setting_name(method, alpha):
    if method == "baseline":
        return "baseline"
    strength = str(abs(float(alpha))).removesuffix(".0").replace(".", "p")
    return f"{method}_{'plus' if alpha > 0 else 'minus'}{strength}sigma"


def make_settings(model_config, generation, method, alpha, layer, sigma, details, prompts):
    method, scope = method_and_scope(method)
    if method != "baseline" and details.get("source_scope", "full") != scope:
        raise ValueError("Method and direction source scope must agree")
    baseline = method == "baseline"
    return {"model_config": model_config, "generation": generation, "method": method,
            "alpha": 0.0 if baseline else alpha, "layer": None if baseline else layer,
            "sigma": None if baseline else sigma, "direction_selection": {} if baseline else details,
            "iteration_order": "sample_prompt", "prompts": prompts}


def read_saved_output(directory, settings, resume):
    if directory.exists() and any(directory.iterdir()) and not resume:
        raise FileExistsError(f"Results already exist in {directory}; use --resume or another --output-dir")
    settings_path, records_path = directory / "settings.json", directory / "generations.jsonl"
    previous = None
    if settings_path.exists():
        previous = json.loads(settings_path.read_text(encoding="utf-8"))
        if {key: previous.get(key) for key in settings} != settings:
            raise ValueError(f"Saved settings differ in {directory}; use another --output-dir")
    elif records_path.exists():
        raise FileNotFoundError(f"Cannot resume {records_path} without settings.json")
    return previous, read_existing(records_path, settings)


def check_saved_runtime(previous, runtime, pad_token_id):
    if previous is None:
        return
    if not isinstance(previous.get("runtime"), dict):
        raise ValueError("Saved outputs have no generation runtime; use a new --output-dir for generation")
    saved = dict(previous["runtime"])
    if "model_generation_config" in saved:
        defaults = saved.pop("model_generation_config")
        saved["effective_generation_config"] = effective_generation_config(
            defaults, previous["generation"], pad_token_id)
    if saved != runtime:
        raise ValueError("Runtime or effective generation settings changed; use a new output directory")


def main():
    artifacts = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Model JSON, e.g. config/qwen.json")
    parser.add_argument("--generation-config", type=Path, default=Path(__file__).parent / "config/generation.json")
    parser.add_argument("--prompts", type=Path, default=artifacts / "data/generation_prompts.jsonl")
    parser.add_argument("--input-dir", type=Path, help="Model outputs containing saved directions and sigma.json")
    parser.add_argument("--output-dir", type=Path, help="Parent folder for named settings; defaults to <input-dir>/generation")
    parser.add_argument("--methods", nargs="+", choices=("baseline", *DIRECTION_FILES,
                                                        *EXCLUDED_METHODS, *COMPARISON_METHODS),
                        default=["baseline", "crypsteer"])
    parser.add_argument("--alpha", type=float, default=1.0, help="Signed strength in sigma units for steered methods")
    parser.add_argument("--samples-per-prompt", type=int, help="Override the configured 30 attempts per prompt")
    parser.add_argument("--limit", type=int, help="First N prompt/seed identities per method; --limit 1 gives one each")
    parser.add_argument("--max-hours", type=float, help="Stop after this generation time, finishing the current paired identity")
    parser.add_argument("--resume", action="store_true", help="Resume existing settings and create missing ones, skipping saved outputs")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if len(set(args.methods)) != len(args.methods):
        parser.error("Each method must appear only once")
    if not math.isfinite(args.alpha) or (args.alpha == 0 and any(m != "baseline" for m in args.methods)):
        parser.error("Use a finite, nonzero steering alpha; select baseline for unsteered generation")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.max_hours is not None and (not math.isfinite(args.max_hours) or args.max_hours <= 0):
        parser.error("--max-hours must be finite and positive")
    model_config = json.loads(args.config.read_text(encoding="utf-8"))
    generation = json.loads(args.generation_config.read_text(encoding="utf-8"))
    if args.samples_per_prompt is not None:
        generation["samples_per_prompt"] = args.samples_per_prompt
    for name in ("samples_per_prompt", "max_new_tokens"):
        if type(generation[name]) is not int or generation[name] < 1:
            parser.error(f"{name} must be a positive integer")
    if type(generation["seed_start"]) is not int or generation["seed_start"] < 0:
        parser.error("seed_start must be a nonnegative integer")
    prompts = load_prompts(args.prompts)
    input_dir = args.input_dir or artifacts / "outputs" / model_config["model_name"]
    output = args.output_dir or input_dir / "generation"
    if (output / "generations.jsonl").exists() or (output / "settings.json").exists():
        raise ValueError(f"{output} contains a flat result file. --output-dir must be a parent of named "
                         "setting folders; choose a new parent for the previous mixed-output test.")

    import numpy as np
    import torch
    import transformers
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    vectors, row_maps, layer, sigma, details = load_directions(input_dir, args.methods, model_config)
    for method in row_maps:
        for prompt in prompts:
            direction_for_prompt(method, prompt, vectors, row_maps)
    directories, settings, previous = {}, {}, {}
    existing = set()
    for method in args.methods:
        directory = output / setting_name(method, args.alpha)
        directories[method] = directory
        settings[method] = make_settings(model_config, generation, method, args.alpha, layer, sigma,
                                         details.get(method, {}), prompts)
        previous[method], saved = read_saved_output(directory, settings[method], args.resume)
        existing.update((method, prompt_id, index) for _, prompt_id, index in saved)
    
    jobs = [(prompt, index) for index in range(generation["samples_per_prompt"]) for prompt in prompts]
    if args.limit is not None:
        jobs = jobs[:args.limit]
    selected_keys = {(method, p["prompt_id"], index) for p, index in jobs for method in args.methods}
    complete = len(selected_keys & existing)
    if complete == len(selected_keys):
        print(f"All {complete} selected outputs already exist under {output}")
        return

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_path"], use_fast=True, trust_remote_code=model_config["trust_remote_code"],
        local_files_only=args.local_files_only)
    tokenizer.padding_side = "left"
    load_options = {"dtype": getattr(torch, model_config["dtype"]),
                    "trust_remote_code": model_config["trust_remote_code"],
                    "local_files_only": args.local_files_only}
    if device == "cuda":
        load_options["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_config["model_path"], **load_options)
    if device == "cpu":
        model.to("cpu")
    model.eval()
    if vectors and (layer >= model.config.num_hidden_layers or any(
            vector.shape[-1] != model.config.hidden_size for vector in vectors.values())):
        raise ValueError("The saved directions do not match the loaded model")
    runtime = {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
               "device": device, "cuda": torch.version.cuda,
               "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if device == "cuda" else [],
               "effective_generation_config": effective_generation_config(
                   model.generation_config.to_dict(), generation, tokenizer.eos_token_id)}
    for method in args.methods:
        check_saved_runtime(previous[method], runtime, tokenizer.eos_token_id)
    for method in args.methods:
        directories[method].mkdir(parents=True, exist_ok=True)
        if previous[method] is None or "model_generation_config" in previous[method]["runtime"]:
            settings[method]["runtime"] = runtime
            (directories[method] / "settings.json").write_text(
                json.dumps(settings[method], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{model_config['model_name']}: {len(jobs)} prompt/seed identities, methods {', '.join(args.methods)}")
    print(f"Layer {layer}, sigma {sigma}, alpha {args.alpha}; batch size 1.")
    start = time.monotonic()
    with tqdm(total=len(selected_keys), initial=complete, desc="Generate programs", unit="output") as progress:
        for prompt, index in jobs:
            if args.max_hours is not None and time.monotonic() - start >= args.max_hours * 3600:
                print("Time budget reached; saved outputs can be continued with --resume.")
                break
            for method in args.methods:
                key = method, prompt["prompt_id"], index
                if key in existing:
                    continue
                baseline = method == "baseline"
                unit = direction_for_prompt(method, prompt, vectors, row_maps)
                coefficient = 0.0 if baseline else args.alpha * sigma
                seed = generation["seed_start"] + index
                attempt_start = time.monotonic()
                raw, input_tokens, output_tokens = generate_one(
                    model, tokenizer, prompt["generation_prompt"], unit, layer, coefficient, seed, generation)
                record = {"model": model_config["model_name"],
                          **{name: prompt[name] for name in ("prompt_id", "language", "misuse_id", "task_id")},
                          "method": settings[method]["method"], "alpha": 0.0 if baseline else args.alpha,
                          "layer": None if baseline else layer, "sigma": None if baseline else sigma,
                          "effective_coefficient": coefficient, "sample_index": index, "seed": seed,
                          "raw_response": raw, "code": extract_program_source(raw, prompt["language"]),
                          "input_tokens": input_tokens, "output_tokens": output_tokens,
                          "elapsed_seconds": time.monotonic() - attempt_start}
                if method in EXCLUDED_METHODS:
                    record["direction_scope"] = EXCLUDED_METHODS[method]
                with (directories[method] / "generations.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                existing.add(key)
                progress.update(1)
    for method in args.methods:
        count = sum((method, prompt["prompt_id"], index) in existing for prompt, index in jobs)
        print(f"{method}: {count}/{len(jobs)} selected outputs in {directories[method] / 'generations.jsonl'}")


if __name__ == "__main__":
    main()
