import argparse
import ast
from itertools import islice
import json
from pathlib import Path
import re


def skip_blank_lines(text, start):
    while start < len(text):
        match = re.match(r"[ \t]*(?:\r?\n|$)", text[start:])
        if match is None or match.end() == 0:
            break
        start += match.end()
    return start


def statement_end(text, start):
    quote, escaped = None, False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
        elif char == "\\" and quote is not None:
            escaped = True
        elif char in {"\"", "'", "`"}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
        elif char == ";" and quote is None:
            return index + 1
    return len(text)


def source_body_span(text, language):
    start = 0
    if language == "java":
        for line in text.splitlines(keepends=True):
            stripped = line.strip()
            if not stripped or stripped.startswith(("package ", "import ")):
                start += len(line)
            else:
                break
        else:
            start = 0
    elif language == "python":
        lines = text.split("\n")
        for node in ast.parse(text).body:
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                break
            if node.end_lineno is not None:
                start = min(len(text), sum(len(line) + 1 for line in lines[:node.end_lineno]))
        start = skip_blank_lines(text, start)
    elif language in {"typescript", "csharp"}:
        start = skip_blank_lines(text, 0)
        while start < len(text):
            end = statement_end(text, start)
            if language == "typescript":
                is_import = re.match(r"import\b", text[start:]) or re.match(
                    r"(?:const|let|var)\s+[^;\n]+?=\s*require\s*\(", text[start:]
                )
            else:
                is_import = re.fullmatch(
                    r"(?:global\s+)?using\s+(?:static\s+)?"
                    r"(?:[A-Za-z_]\w*\s*=\s*)?(?:global::)?"
                    r"[A-Za-z_]\w*(?:(?:::|\.)[A-Za-z_]\w*)*\s*;",
                    text[start:end].strip(),
                )
            if not is_import:
                break
            start = skip_blank_lines(text, end)
    else:
        raise ValueError(f"Unsupported language: {language}")
    end = len(text.rstrip())
    if start >= end:
        raise ValueError(f"No program body found for {language}")
    return start, end


def body_token_indices(offsets, body_start, body_end):
    return [
        t for t, (start, end) in enumerate(offsets)
        if start != end and end > body_start and start < body_end
    ]


def representation_group(record):
    if "original_id" in record:
        return "program"
    if "secure_bits" in record:
        return "combination"
    if "pair_id" in record:
        return "heatmap_program"
    raise ValueError("Expected a matched, combination, or heatmap program record")


def model_input(record):
    text = record["code"].replace("\r\n", "\n").replace("\r", "\n")
    if representation_group(record) != "heatmap_program":
        text = text.rstrip()
    return text


def load_programs(path, limit):
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1")
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in islice(handle, limit)]
    if not records:
        raise ValueError("The selected input contains no programs")
    ids = set()
    for record in records:
        if record["id"] in ids:
            raise ValueError(f"Duplicate program ID: {record['id']}")
        ids.add(record["id"])
        if record["language"] not in {"java", "typescript", "csharp", "python"}:
            raise ValueError(f"Unsupported language in {record['id']}")
        if not isinstance(record["code"], str) or not record["code"].strip():
            raise ValueError(f"Empty source code in {record['id']}")
    if len({representation_group(record) for record in records}) != 1:
        raise ValueError("Use one dataset per extraction run")
    return records


def pool_program_representation(h, I_body):
    if len(I_body) == 1:
        return h[0, I_body[0], :]
    return h[0, I_body, :].mean(dim=0)


def main():
    artifacts = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=artifacts / "data/programs.jsonl")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=artifacts / "outputs")
    parser.add_argument("--limit", type=int, help="Process the first N records in file order")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--local-files-only", action="store_true", help="Use local model files only")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    records = load_programs(args.data, args.limit)
    group = representation_group(records[0])
    model_name = config["model_name"]
    model_dir = args.output_dir / model_name
    representation_dir = model_dir / f"{group}_representations"
    index_path = model_dir / f"{group}_index.jsonl"
    if index_path.exists() or (representation_dir.exists() and any(representation_dir.iterdir())):
        raise FileExistsError(f"Results already exist in {model_dir}; choose another --output-dir")

    import numpy as np
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_path"], use_fast=True,
        trust_remote_code=config["trust_remote_code"],
        local_files_only=args.local_files_only,
    )
    if not tokenizer.is_fast:
        raise ValueError("Extraction requires a fast tokenizer with character offsets")
    tokenizer.padding_side = "left"
    load_options = {
        "dtype": getattr(torch, config["dtype"]),
        "trust_remote_code": config["trust_remote_code"],
        "local_files_only": args.local_files_only,
    }
    if device == "cuda":
        load_options["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(config["model_path"], **load_options)
    if device == "cpu":
        model.to("cpu")
    model.eval()
    input_device = model.get_input_embeddings().weight.device
    layer_count = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    representation_dir.mkdir(parents=True, exist_ok=True)

    arrays = [
        np.lib.format.open_memmap(
            representation_dir / f"layer_{ell:02d}.npy", mode="w+", dtype=np.float32,
            shape=(len(records), hidden_size),
        )
        for ell in range(layer_count)
    ]
    with index_path.open("w", encoding="utf-8") as index_file, torch.inference_mode():
        for row, record in enumerate(tqdm(records, desc=f"{model_name}: h_bar", unit="program")):
            text = model_input(record)
            body_start, body_end = source_body_span(text, record["language"])
            inputs = tokenizer(
                [text], return_tensors="pt", return_offsets_mapping=True,
                add_special_tokens=True, truncation=False,
            )
            offsets = inputs.pop("offset_mapping")[0].tolist()
            I_body = body_token_indices(offsets, body_start, body_end)
            if not I_body:
                raise ValueError(f"No code-body tokens for {record['id']}")
            inputs = {key: value.to(input_device) for key, value in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            if outputs.hidden_states is None or len(outputs.hidden_states) != layer_count + 1:
                raise ValueError("Model hidden-state count does not match its decoder layers")
            for ell, h in enumerate(outputs.hidden_states[1:]):
                h_bar = pool_program_representation(h, I_body)
                arrays[ell][row] = h_bar.float().cpu().numpy()
            index = {key: value for key, value in record.items() if key != "code"}
            index.update({
                "row": row,
                "body_start": body_start,
                "body_end": body_end,
                "body_token_count": len(I_body),
                "input_token_count": len(offsets),
            })
            index_file.write(json.dumps(index, ensure_ascii=False) + "\n")
            del outputs, inputs, h, h_bar
    for array in arrays:
        array.flush()
    print(f"Saved {layer_count} layers, each with shape ({len(records)}, {hidden_size}).")
    print(f"Representations: {representation_dir}")
    print(f"Row index: {index_path}")


if __name__ == "__main__":
    main()
