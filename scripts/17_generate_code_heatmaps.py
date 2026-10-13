"""Project complete paired programs onto CrypSteer and create code heatmaps."""

from bisect import bisect_right
from difflib import SequenceMatcher
import argparse
import html
import json
import math
from pathlib import Path
import re


_TOKEN_RE = re.compile(
    r"(?P<block_comment>/\*.*?\*/)"
    r"|(?P<line_comment>//[^\r\n]*|\#[^\r\n]*)"
    r"|(?P<template>`(?:\\.|[^`\\])*`)"
    r'|(?P<double_string>"(?:\\.|[^"\\])*")'
    r"|(?P<single_string>'(?:\\.|[^'\\])*')"
    r"|(?P<identifier>[A-Za-z_$][A-Za-z0-9_$]*)"
    r"|(?P<number>(?:0[xX][0-9A-Fa-f]+)|(?:\d+(?:\.\d+)?(?:[eE][+-]?\d+)?))"
    r"|(?P<operator>===|!==|>>>|<<=|>>=|=>|::|\?\?|\?\.|==|!=|<=|>=|\+\+|--|&&|\|\||<<|>>|\+=|-=|\*=|/=|%=|&=|\|=|\^=|->)"
    r"|(?P<punctuation>[^\s])"
    r"|(?P<whitespace>\s+)",
    flags=re.DOTALL,
)


def align_source_units(secure_code, misuse_code, language):
    """Match stripped lines first, then lexical tokens within each line block."""
    if language not in {"java", "typescript", "csharp", "python"}:
        raise ValueError(f"Unsupported source language: {language}")
    secure_lines = _source_lines(secure_code)
    misuse_lines = _source_lines(misuse_code)
    secure_starts = [line["start"] for line in secure_lines]
    misuse_starts = [line["start"] for line in misuse_lines]
    matcher = SequenceMatcher(
        None,
        [line["normalized"] for line in secure_lines],
        [line["normalized"] for line in misuse_lines],
        autojunk=False,
    )
    units = []
    for opcode, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if opcode == "equal":
            for offset in range(left_end - left_start):
                units.extend(_align_region_lexemes(
                    secure_code, misuse_code,
                    secure_start=secure_lines[left_start + offset]["start"],
                    secure_end=secure_lines[left_start + offset]["end"],
                    misuse_start=misuse_lines[right_start + offset]["start"],
                    misuse_end=misuse_lines[right_start + offset]["end"],
                    secure_line_starts=secure_starts,
                    misuse_line_starts=misuse_starts,
                ))
            continue
        secure_span = _line_block_span(secure_lines, left_start, left_end)
        misuse_span = _line_block_span(misuse_lines, right_start, right_end)
        if opcode == "replace":
            units.extend(_align_region_lexemes(
                secure_code, misuse_code,
                secure_start=secure_span[0], secure_end=secure_span[1],
                misuse_start=misuse_span[0], misuse_end=misuse_span[1],
                secure_line_starts=secure_starts,
                misuse_line_starts=misuse_starts,
            ))
        elif opcode == "delete":
            units.extend(_unpaired_region(
                secure_code, start=secure_span[0], end=secure_span[1],
                line_starts=secure_starts, secure_side=True,
            ))
        elif opcode == "insert":
            units.extend(_unpaired_region(
                misuse_code, start=misuse_span[0], end=misuse_span[1],
                line_starts=misuse_starts, secure_side=False,
            ))
    for index, unit in enumerate(units):
        unit["unit_id"] = index
    return units


def score_alignment_units(units, secure_token_rows, misuse_token_rows):
    """Average token projections over each span and subtract misuse from secure."""
    output = []
    for unit in units:
        secure_projection = _span_score(
            secure_token_rows, unit["secure_start"], unit["secure_end"]
        )
        misuse_projection = _span_score(
            misuse_token_rows, unit["misuse_start"], unit["misuse_end"]
        )
        delta = (
            secure_projection - misuse_projection
            if secure_projection is not None and misuse_projection is not None
            else None
        )
        output.append({
            **unit,
            "secure_projection": secure_projection,
            "misuse_projection": misuse_projection,
            "projection_delta": delta,
        })
    return output


def _span_score(rows, start, end):
    if start is None or end is None or end <= start:
        return None
    weighted_sum = 0.0
    total_weight = 0
    for row in rows:
        token_start, token_end = int(row["char_start"]), int(row["char_end"])
        overlap = min(end, token_end) - max(start, token_start)
        if overlap <= 0:
            continue
        weighted_sum += float(row["raw_score"]) * overlap
        total_weight += overlap
    return weighted_sum / total_weight if total_weight else None


def _source_lines(code):
    lines = []
    cursor = 0
    for number, text in enumerate(code.splitlines(keepends=True), start=1):
        end = cursor + len(text)
        lines.append({"number": number, "start": cursor, "end": end,
                      "text": text, "normalized": text.strip()})
        cursor = end
    if cursor < len(code):
        text = code[cursor:]
        lines.append({"number": len(lines) + 1, "start": cursor, "end": len(code),
                      "text": text, "normalized": text.strip()})
    return lines


def _line_block_span(lines, start, end):
    if start == end:
        position = lines[start]["start"] if start < len(lines) else (lines[-1]["end"] if lines else 0)
        return position, position
    return lines[start]["start"], lines[end - 1]["end"]


def _lexemes(code, *, start, end, line_starts):
    result = []
    for match in _TOKEN_RE.finditer(code, start, end):
        if match.start() >= end:
            break
        kind = str(match.lastgroup)
        if kind in {"whitespace", "line_comment", "block_comment"}:
            continue
        result.append({
            "kind": kind,
            "text": match.group(0),
            "start": match.start(),
            "end": min(match.end(), end),
            "line": bisect_right(line_starts, match.start()),
        })
    return result


def _alignment_unit(kind, secure_start, secure_end, misuse_start, misuse_end,
                    secure_line, misuse_line, secure_text, misuse_text):
    return {
        "unit_id": -1,
        "kind": kind,
        "secure_start": secure_start,
        "secure_end": secure_end,
        "misuse_start": misuse_start,
        "misuse_end": misuse_end,
        "secure_line": secure_line,
        "misuse_line": misuse_line,
        "secure_text": secure_text,
        "misuse_text": misuse_text,
        "alignment_label": None,
    }


def _align_region_lexemes(secure_code, misuse_code, *, secure_start, secure_end,
                         misuse_start, misuse_end, secure_line_starts, misuse_line_starts):
    secure = _lexemes(secure_code, start=secure_start, end=secure_end,
                      line_starts=secure_line_starts)
    misuse = _lexemes(misuse_code, start=misuse_start, end=misuse_end,
                      line_starts=misuse_line_starts)
    matcher = SequenceMatcher(
        None,
        [(token["kind"], token["text"]) for token in secure],
        [(token["kind"], token["text"]) for token in misuse],
        autojunk=False,
    )
    result = []
    for opcode, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if opcode == "equal":
            for offset in range(left_end - left_start):
                left = secure[left_start + offset]
                right = misuse[right_start + offset]
                result.append(_alignment_unit(
                    "matched_token", left["start"], left["end"], right["start"], right["end"],
                    left["line"], right["line"], left["text"], right["text"],
                ))
            continue
        left_tokens = secure[left_start:left_end]
        right_tokens = misuse[right_start:right_end]
        if left_tokens and right_tokens:
            result.append(_alignment_unit(
                "changed_region",
                left_tokens[0]["start"], left_tokens[-1]["end"],
                right_tokens[0]["start"], right_tokens[-1]["end"],
                left_tokens[0]["line"], right_tokens[0]["line"],
                secure_code[left_tokens[0]["start"]:left_tokens[-1]["end"]],
                misuse_code[right_tokens[0]["start"]:right_tokens[-1]["end"]],
            ))
        elif left_tokens:
            result.append(_tokens_to_unpaired(left_tokens, secure_code, secure_side=True))
        elif right_tokens:
            result.append(_tokens_to_unpaired(right_tokens, misuse_code, secure_side=False))
    return result


def _unpaired_region(code, *, start, end, line_starts, secure_side):
    tokens = _lexemes(code, start=start, end=end, line_starts=line_starts)
    return [] if not tokens else [_tokens_to_unpaired(tokens, code, secure_side=secure_side)]


def _tokens_to_unpaired(tokens, code, *, secure_side):
    start, end = tokens[0]["start"], tokens[-1]["end"]
    text = code[start:end]
    return _alignment_unit(
        "unpaired_secure" if secure_side else "unpaired_misuse",
        start if secure_side else None,
        end if secure_side else None,
        None if secure_side else start,
        None if secure_side else end,
        tokens[0]["line"] if secure_side else None,
        None if secure_side else tokens[0]["line"],
        text if secure_side else "",
        "" if secure_side else text,
    )


def load_pairs(path, limit=None):
    """Keep file order, normalize newlines, and retain complete program pairs."""
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1 complete pair")
    pairs, ids = {}, set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["id"] in ids:
                raise ValueError(f"Duplicate program ID: {record['id']}")
            ids.add(record["id"])
            if (record["language"] not in {"java", "typescript", "csharp", "python"}
                    or record["misuse_id"] not in {f"M{i:02d}" for i in range(1, 11)}
                    or record["label"] not in {"secure", "misuse"}
                    or not isinstance(record["code"], str) or not record["code"].strip()):
                raise ValueError(f"Invalid heatmap program: {record['id']}")
            pair = pairs.setdefault(record["pair_id"], {
                key: record[key] for key in ("pair_id", "language", "misuse_id")})
            if any(pair[key] != record[key] for key in ("language", "misuse_id")):
                raise ValueError(f"Pair metadata differs: {record['pair_id']}")
            if record["label"] in pair:
                raise ValueError(f"Duplicate label in pair: {record['pair_id']}")
            # Match the old text-file reader; the final newline is part of the input.
            pair[record["label"]] = {**record, "code": record["code"].replace("\r\n", "\n").replace("\r", "\n")}
    rows = list(pairs.values())
    if not rows or any("secure" not in pair or "misuse" not in pair for pair in rows):
        raise ValueError("Each heatmap pair needs one secure and one misuse program")
    if len({(p["language"], p["misuse_id"]) for p in rows}) != len(rows):
        raise ValueError("Expected one heatmap pair per language and misuse category")
    return rows[:limit], len(rows)


def source_token_score_rows(source, offsets, h, v_CrypSteer):
    """Use the original float32 CPU projection, with no token normalization."""
    import numpy as np

    if hasattr(h, "detach"):
        h = h.detach().cpu().float().numpy()
    states = np.asarray(h, dtype=np.float32)
    direction = np.asarray(v_CrypSteer, dtype=np.float32)
    if states.ndim != 2 or direction.shape != (states.shape[1],) or len(offsets) != len(states):
        raise ValueError("Token activations, offsets and direction have incompatible shapes")
    scores = (states @ direction).astype(np.float32)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite token projection")
    rows = []
    for index, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        if start < 0 or end > len(source):
            raise ValueError("Tokenizer offset falls outside the source")
        rows.append({"token_index": index, "char_start": int(start), "char_end": int(end),
                     "token_text": source[start:end], "raw_score": float(scores[index])})
    if not rows:
        raise ValueError("Tokenizer produced no source-overlapping tokens")
    return rows


def project_program(model, tokenizer, code, layer, v_CrypSteer):
    import torch

    inputs = tokenizer([code], return_tensors="pt", return_offsets_mapping=True,
                       add_special_tokens=True, truncation=False, padding=False)
    offsets = inputs.pop("offset_mapping")[0].tolist()
    device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        result = model(**inputs, output_hidden_states=True, use_cache=False)
    if result.hidden_states is None or len(result.hidden_states) != model.config.num_hidden_layers + 1:
        raise ValueError("Unexpected model hidden-state count")
    # Entry zero is the embedding; subsequent entries follow decoder-block indices.
    return source_token_score_rows(code, offsets, result.hidden_states[layer + 1][0], v_CrypSteer)


PAGE_STYLE = """
*{box-sizing:border-box}
body{margin:0;background:#f6f7f9;color:#1f2937;font:14px/1.5 Arial,sans-serif}
main{max-width:1800px;margin:auto;padding:20px}h1{font-size:21px;margin:0}
.topline{display:flex;justify-content:space-between;align-items:center;gap:16px;margin-bottom:12px}
.topline a{font-size:13px;white-space:nowrap;text-decoration:none}
h2{font-size:17px;margin:0;padding:12px 16px;border-bottom:1px solid #dce1e6}
a{color:#315f8c}.meta{color:#51606c;font-size:12px}
.toolbar{position:sticky;top:0;z-index:5;background:rgba(255,255,255,.97);
border:1px solid #d8dee8;padding:12px;box-shadow:0 2px 8px rgba(31,41,55,.06)}
.controls{display:grid;grid-template-columns:repeat(2,minmax(0,1fr)) auto;
gap:12px 24px;align-items:center}
.control{display:grid;gap:5px;min-width:0;font-size:12px;color:#475467}
.control input[type=range]{width:100%;margin:0;accent-color:#315f8c;cursor:pointer}
.control output{font:12px Menlo,Consolas,monospace;color:#1f2937}
button{font:inherit;padding:5px 12px;border:1px solid #cdd5df;background:#fff;cursor:pointer}
.panels{display:grid;
grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-top:16px}
section{background:white;border:1px solid #dce1e6;border-radius:8px;overflow:hidden}
pre{margin:0;padding:16px;overflow:auto;font:13px/1.65 ui-monospace,Consolas,monospace;
tab-size:4;white-space:pre}code{font:inherit}.heat{border-radius:2px;cursor:help}
.unpaired{background:#ededed;text-decoration:underline dotted #888}
.counterpart{outline:2px solid #527c98;outline-offset:1px}
.score-tooltip{position:fixed;z-index:50;max-width:min(420px,calc(100vw - 16px));
max-height:calc(100vh - 16px);overflow:hidden;padding:8px 10px;background:#101828;
color:#fff;border-radius:4px;box-shadow:0 8px 22px rgba(16,24,40,.24);
white-space:pre-line;overflow-wrap:anywhere;font:12px/1.45 Menlo,Consolas,monospace;
pointer-events:none}
.score-tooltip[hidden]{display:none}
table{border-collapse:collapse;background:white}th,td{text-align:left;padding:9px 20px;
border-bottom:1px solid #dce1e6}
@media(max-width:800px){.panels{grid-template-columns:1fr}main{padding:12px}
.controls{grid-template-columns:1fr}.topline{align-items:flex-start}}
"""


def pair_data(pair, secure_tokens, misuse_tokens, spans):
    return {**{k: pair[k] for k in ("pair_id", "language", "misuse_id")},
            "secure": {**pair["secure"], "tokens": secure_tokens},
            "misuse": {**pair["misuse"], "tokens": misuse_tokens}, "spans": spans}


def render_source(source, spans, side):
    """Escape source text while keeping all original whitespace and comments."""
    fragments, cursor = [], 0
    units = sorted((s for s in spans if s[f"{side}_start"] is not None),
                   key=lambda s: s[f"{side}_start"])
    for span in units:
        start, end = span[f"{side}_start"], span[f"{side}_end"]
        if not 0 <= cursor <= start < end <= len(source):
            raise ValueError("Overlapping or invalid aligned source spans")
        fragments.append(html.escape(source[cursor:start]))
        delta = span["projection_delta"]
        signed = delta if side == "secure" or delta is None else -delta
        describe = lambda value: "unmatched" if value is None else format(value, ".8g")
        tooltip = (f"Secure projection: {describe(span['secure_projection'])}\n"
                 f"Misuse projection: {describe(span['misuse_projection'])}\n"
                 f"Delta (secure - misuse): {describe(delta)}\n"
                 f"Displayed value: {describe(signed)}\n"
                 f"Alignment: {span['kind']}")
        attributes = f'data-unit="{span["unit_id"]}" data-tooltip="{html.escape(tooltip, quote=True)}"'
        if delta is not None:
            attributes += f' data-delta="{delta!r}" data-score="{signed!r}"'
        classes = "heat" if delta is not None else "heat unpaired"
        fragments.append(f'<span class="{classes}" {attributes}>{html.escape(source[start:end])}</span>')
        cursor = end
    fragments.append(html.escape(source[cursor:]))
    return "".join(fragments)


def render_pair_html(pair, secure_tokens, misuse_tokens, spans, model_name, layer,
                     minimum_delta=0.2, color_range=0.5):
    if not math.isfinite(minimum_delta) or minimum_delta < 0 or not math.isfinite(color_range) or color_range <= 0:
        raise ValueError("Minimum delta must be nonnegative and color range positive")
    title = html.escape(f"{model_name} · {pair['language']} {pair['misuse_id']}")
    data = json.dumps(pair_data(pair, secure_tokens, misuse_tokens, spans), ensure_ascii=False, allow_nan=False)
    # JSON stays inert even when a code string contains an HTML closing tag.
    data = data.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    panels = []
    for side, heading in (("secure", "Secure (+Δ)"), ("misuse", "Misuse (−Δ)")):
        code = render_source(pair[side]["code"], spans, side)
        panels.append(f'<section><h2>{heading}</h2><pre><code>{code}</code></pre></section>')
    scale_min = min(0.01, color_range / 50)
    scale_max = max(color_range * 4, minimum_delta * 2)
    threshold_max = max(color_range, minimum_delta)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · CrypSteer</title><style>{PAGE_STYLE}</style></head><body><main>
<div class="topline"><div><h1>{title}</h1>
<span class="meta">CrypSteer · Layer {layer} · Δ = secure − misuse</span></div>
<a href="index.html">All pairs</a></div>
<div class="toolbar"><div class="controls">
<label class="control"><span>Color range ± <output id="color-range-value" for="color-range">{color_range:.4f}</output></span>
<input id="color-range" type="range" min="{scale_min!r}" max="{scale_max!r}" step="any" value="{color_range!r}"></label>
<label class="control"><span>Minimum |delta| <output id="minimum-delta-value" for="minimum-delta">{minimum_delta:.4f}</output></span>
<input id="minimum-delta" type="range" min="0" max="{threshold_max!r}" step="any" value="{minimum_delta!r}"></label>
<button id="reset" type="button">Reset</button>
</div></div>
<div class="panels">{''.join(panels)}</div>
<div id="score-tooltip" class="score-tooltip" role="tooltip" hidden></div>
<script id="heatmap-data" type="application/json">{data}</script>
<script>
const minimum = document.getElementById('minimum-delta');
const range = document.getElementById('color-range');
const minimumValue = document.getElementById('minimum-delta-value');
const rangeValue = document.getElementById('color-range-value');
const spans = Array.from(document.querySelectorAll('.heat[data-score]'));
const tooltip = document.getElementById('score-tooltip');
let hoveredSpan = null, counterparts = [];
function hideTooltip() {{
  tooltip.hidden = true;
  counterparts.forEach(s => s.classList.remove('counterpart'));
  hoveredSpan = null; counterparts = [];
}}
function showTooltip(span, event) {{
  if (hoveredSpan !== span) {{
    hideTooltip();
    hoveredSpan = span;
    counterparts = Array.from(document.querySelectorAll(`[data-unit="${{span.dataset.unit}}"]`));
    counterparts.forEach(s => s.classList.add('counterpart'));
    tooltip.textContent = span.dataset.tooltip;
  }}
  tooltip.hidden = false;
  tooltip.style.left = '0px'; tooltip.style.top = '0px';
  const box = tooltip.getBoundingClientRect();
  const x = Math.max(8, Math.min(event.clientX + 14, window.innerWidth - box.width - 8));
  const below = event.clientY + 14;
  const y = Math.max(8, below + box.height <= window.innerHeight - 8
    ? below : event.clientY - box.height - 14);
  tooltip.style.left = `${{x}}px`;
  tooltip.style.top = `${{y}}px`;
}}
function paint() {{
  const threshold = Number(minimum.value), scale = Number(range.value);
  minimumValue.value = threshold.toFixed(4);
  rangeValue.value = scale.toFixed(4);
  for (const span of spans) {{
    const value = Number(span.dataset.score);
    const active = Math.abs(value) > threshold;
    const rgb = value >= 0 ? '28,145,78' : '205,49,49';
    const opacity = 0.82 * Math.min(Math.abs(value) / scale, 1);
    span.style.backgroundColor = active ? `rgba(${{rgb}},${{opacity}})` : 'transparent';
  }}
}}
function resizeRange() {{
  const threshold = Math.min(Number(minimum.value), Number(range.value));
  minimum.max = range.value;
  minimum.value = String(threshold);
  paint();
}}
for (const event of ['input', 'change']) {{
  minimum.addEventListener(event, paint);
  range.addEventListener(event, resizeRange);
}}
document.getElementById('reset').addEventListener('click', () => {{
  range.value = '{color_range!r}'; minimum.max = '{threshold_max!r}';
  minimum.value = '{minimum_delta!r}'; paint();
}});
for (const span of document.querySelectorAll('.heat')) {{
  span.addEventListener('mouseenter', event => showTooltip(span, event));
  span.addEventListener('mousemove', event => showTooltip(span, event));
  span.addEventListener('mouseleave', hideTooltip);
}}
document.addEventListener('scroll', hideTooltip, true);
window.addEventListener('resize', hideTooltip);
window.addEventListener('blur', hideTooltip);
document.addEventListener('keydown', event => {{ if (event.key === 'Escape') hideTooltip(); }});
paint();
</script></main></body></html>
'''


def render_index(pairs, model_name, layer):
    rows = []
    for pair in pairs:
        filename = f"{pair['language']}__{pair['misuse_id']}.html"
        rows.append(f'<tr><td>{html.escape(pair["language"])}</td><td>{html.escape(pair["misuse_id"])}</td>'
                    f'<td><a href="{html.escape(filename, quote=True)}">Open paired heatmap</a></td></tr>')
    title = html.escape(f"{model_name} · CrypSteer code heatmaps")
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>{PAGE_STYLE}</style></head><body><main><h1>{title}</h1>
<p class="meta">Layer {layer} · {len(pairs)} pairs</p>
<table><thead><tr><th>Language</th><th>Misuse category</th><th>Programs</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></main></body></html>'''


def render_saved_heatmaps(folder, minimum_delta=0.2, color_range=0.5):
    """Refresh only the HTML, using the saved projections and alignments."""
    from tqdm import tqdm

    settings = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
    with (folder / "heatmaps.jsonl").open(encoding="utf-8") as handle:
        pairs = [json.loads(line) for line in handle if line.strip()]
    model_name, layer = settings["model_config"]["model_name"], settings["layer"]
    for pair in tqdm(pairs, desc="Render heatmaps", unit="pair"):
        page = render_pair_html(pair, pair["secure"]["tokens"], pair["misuse"]["tokens"],
                                pair["spans"], model_name, layer, minimum_delta, color_range)
        (folder / f"{pair['language']}__{pair['misuse_id']}.html").write_text(page, encoding="utf-8")
    (folder / "index.html").write_text(render_index(pairs, model_name, layer), encoding="utf-8")
    print(f"Updated HTML: {folder / 'index.html'}")


def main():
    artifacts = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Model JSON; required for activation extraction")
    parser.add_argument("--render-only", type=Path, metavar="FOLDER",
                        help="Refresh HTML in an existing code_heatmaps folder without loading a model")
    parser.add_argument("--data", type=Path, default=artifacts / "data/heatmap_programs.jsonl")
    parser.add_argument("--direction-dir", type=Path, help="Step 04 directions/; defaults to outputs/MODEL/directions")
    parser.add_argument("--output-dir", type=Path, default=artifacts / "outputs",
                        help="Parent folder; saves MODEL/code_heatmaps/")
    parser.add_argument("--limit", type=int, help="First N complete secure/misuse pairs in data-file order")
    parser.add_argument("--minimum-delta", type=float, default=0.2, help="Initial display threshold, adjustable in HTML")
    parser.add_argument("--color-range", type=float, default=0.5, help="Initial symmetric color range, adjustable in HTML")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if (not math.isfinite(args.minimum_delta) or args.minimum_delta < 0
            or not math.isfinite(args.color_range) or args.color_range <= 0):
        parser.error("Minimum delta must be nonnegative and color range positive")
    if args.render_only is not None:
        render_saved_heatmaps(args.render_only, args.minimum_delta, args.color_range)
        return
    if args.config is None:
        parser.error("--config is required unless --render-only is used")
    pairs, population = load_pairs(args.data, args.limit)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    direction_dir = args.direction_dir or artifacts / "outputs" / config["model_name"] / "directions"
    selection = json.loads((direction_dir / "selection.json").read_text(encoding="utf-8"))
    layer = selection["layer"]
    if type(layer) is not int or layer < 0 or selection.get("full_source") is not True:
        raise ValueError("Use a full-source CrypSteer direction from step 04 with its selected layer")
    output = args.output_dir / config["model_name"] / "code_heatmaps"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Results already exist in {output}; choose another --output-dir")

    import numpy as np
    import torch
    import transformers
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    v_CrypSteer = np.load(direction_dir / "v_CrypSteer.npy", allow_pickle=False)
    if (v_CrypSteer.ndim != 1 or v_CrypSteer.dtype != np.float32 or not v_CrypSteer.size
            or not np.isfinite(v_CrypSteer).all()
            or not np.isclose(np.linalg.norm(v_CrypSteer), 1.0, atol=2e-6, rtol=0)):
        raise ValueError("Expected step 04's saved float32 unit direction")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_path"], use_fast=True, trust_remote_code=config["trust_remote_code"],
        local_files_only=args.local_files_only)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer with character offsets is required")
    tokenizer.padding_side = "left"
    options = {"dtype": getattr(torch, config["dtype"]), "trust_remote_code": config["trust_remote_code"],
               "local_files_only": args.local_files_only}
    if device == "cuda":
        options["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(config["model_path"], **options)
    if device == "cpu":
        model.to("cpu")
    model.eval()
    if layer >= model.config.num_hidden_layers or v_CrypSteer.shape != (model.config.hidden_size,):
        raise ValueError("Direction layer or width does not match the loaded model")

    output.mkdir(parents=True, exist_ok=True)
    with (output / "heatmaps.jsonl").open("w", encoding="utf-8") as handle:
        for pair in tqdm(pairs, desc=f"{config['model_name']}: code heatmaps", unit="pair"):
            secure = project_program(model, tokenizer, pair["secure"]["code"], layer, v_CrypSteer)
            misuse = project_program(model, tokenizer, pair["misuse"]["code"], layer, v_CrypSteer)
            units = align_source_units(pair["secure"]["code"], pair["misuse"]["code"], pair["language"])
            spans = score_alignment_units(units, secure, misuse)
            handle.write(json.dumps(pair_data(pair, secure, misuse, spans), ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            page = render_pair_html(pair, secure, misuse, spans, config["model_name"], layer,
                                    args.minimum_delta, args.color_range)
            (output / f"{pair['language']}__{pair['misuse_id']}.html").write_text(page, encoding="utf-8")
    settings = {
        "model_config": config, "layer": layer, "direction": "CrypSteer",
        "selected_pc": selection["selected_pc"], "selected_sign": selection["selected_sign"],
        "direction_full_source": True, "pairs": len(pairs), "programs": 2 * len(pairs),
        "population_pairs": population, "limit_pairs": args.limit,
        "full_corpus": len(pairs) == population,
        "input": "complete source; normalized line endings; final newline retained; no chat template",
        "add_special_tokens": True, "truncation": False, "batch_size": 1,
        "projection": "float32 CPU token activation dot saved unit direction; no additional normalization or sigma",
        "span_projection": "token projections weighted by character overlap",
        "delta": "secure minus misuse; unmatched spans remain null",
        "display": {"minimum_delta": args.minimum_delta, "color_range": args.color_range},
        "runtime": {"torch": torch.__version__, "transformers": transformers.__version__,
                    "numpy": np.__version__, "device": device, "cuda": torch.version.cuda,
                    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                    if device == "cuda" else []},
    }
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    (output / "index.html").write_text(render_index(pairs, config["model_name"], layer), encoding="utf-8")
    print(f"Saved {len(pairs)} paired heatmaps: {output / 'index.html'}")
    print(f"Token projections and aligned spans: {output / 'heatmaps.jsonl'}")


if __name__ == "__main__":
    main()
