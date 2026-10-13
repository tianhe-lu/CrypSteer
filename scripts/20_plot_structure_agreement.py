"""Plot cross-structure cosine and raw repair-norm histograms for one cell."""

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
from tqdm import tqdm


LANGUAGES = {"java": "Java", "typescript": "TypeScript", "csharp": "C#", "python": "Python"}
MISUSES = tuple(f"M{i:02d}" for i in range(1, 11))
MODEL_NAMES = {
    "llama": "Llama", "codegemma": "CodeGemma", "qwen": "Qwen",
    "qwen_coder": "Qwen-Coder", "deepseek": "DeepSeek",
}
STRUCTURES = ("callsite_helper", "callsite_inline", "variable_helper", "variable_inline")
PAIRS = tuple(combinations(range(4), 2))
COLORS = ("#0072B2", "#E69F00", "#7A5195", "#CC79A7")
LINESTYLES = ("-", "--", "-.", ":")


def load_cell(model_dir, language, misuse, limit):
    folder = model_dir / "structure_agreement"
    with (folder / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    cells = [cell for cell in summary["by_cell"]
             if (cell["language"], cell["misuse_id"]) == (language, misuse)]
    if len(cells) != 1:
        raise ValueError(f"{folder}: expected one summary for {language}/{misuse}")
    with (folder / "agreement.csv").open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if (row["language"], row["misuse_id"]) == (language, misuse)]
    if (not rows or len(rows) != cells[0]["originals"]
            or len({row["original_id"] for row in rows}) != len(rows)):
        raise ValueError(f"{folder}: missing or duplicate originals for {language}/{misuse}")
    rows.sort(key=lambda row: (row["task_id"], row["sample_id"], row["original_id"]))
    if limit is not None and limit > len(rows):
        raise ValueError(f"--limit {limit} exceeds the {len(rows)} available originals")
    rows = rows[:limit]
    # These are the six unordered structure pairs within each original.
    cosines = np.asarray([
        [float(row[f"cosine_{STRUCTURES[a]}__{STRUCTURES[b]}"]) for a, b in PAIRS]
        for row in rows
    ], dtype=np.float64)
    norms = np.asarray([
        [float(row[f"norm_{structure}"]) for structure in STRUCTURES] for row in rows
    ], dtype=np.float64)
    if not np.isfinite(cosines).all() or np.any(np.abs(cosines) > 1):
        raise ValueError(f"{folder}: cosines must be finite and lie in [-1, 1]")
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError(f"{folder}: raw repair norms must be positive and finite")
    return cosines, norms, summary["layer"]


def plot_agreement(cosines, norms, title, bins):
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7,
        "axes.linewidth": 0.5, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.7 / 2.54, 4.5 / 2.54))
    fig.subplots_adjust(left=0.15, right=0.985, top=0.69, bottom=0.17, wspace=0.66)
    fig.suptitle(title, fontsize=8, y=0.985)

    # Shared bins across the four structures; keep every observed value in range.
    cosine_lower = max(-1.0, float(np.floor(cosines.min() * 100) / 100))
    if cosine_lower >= 1:
        cosine_lower = 0.99
    cosine_bins = np.linspace(cosine_lower, 1, bins + 1)
    norm_bins = np.linspace(0, float(norms.max()) * 1.001, bins + 1)
    for structure_index, structure in enumerate(tqdm(
            STRUCTURES, desc="Plot structure agreement", unit="structure")):
        pair_columns = [j for j, pair in enumerate(PAIRS) if structure_index in pair]
        matched_cosines = cosines[:, pair_columns].reshape(-1)
        # Each structure has 3*N cosines and N norms. Each unordered pair appears
        # in two curves. Bin proportions sum to one separately for every curve.
        axes[0].hist(
            matched_cosines, bins=cosine_bins,
            weights=np.full(matched_cosines.size, 1 / matched_cosines.size),
            density=False, histtype="step", linewidth=0.85,
            linestyle=LINESTYLES[structure_index], color=COLORS[structure_index],
            label=structure.replace("_", "/"),
        )
        axes[1].hist(
            norms[:, structure_index], bins=norm_bins,
            weights=np.full(len(norms), 1 / len(norms)),
            density=False, histtype="step", linewidth=0.85,
            linestyle=LINESTYLES[structure_index], color=COLORS[structure_index],
        )
    axes[0].set(xlim=(cosine_lower, 1), xlabel="Cosine", ylabel="Proportion")
    axes[1].set(xlim=(0, norm_bins[-1]), xlabel="Raw repair norm", ylabel="Proportion")
    # Panel letters and sample counts belong in the LaTeX caption.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, fontsize=6.5, frameon=False,
               loc="upper center", bbox_to_anchor=(0.53, 0.92),
               handlelength=1.8, columnspacing=1.1, labelspacing=0.3, borderaxespad=0)
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
        ax.tick_params(labelsize=6.5, length=2, width=0.5, pad=1.5)
        ax.xaxis.label.set_size(7)
        ax.yaxis.label.set_size(7)
        ax.xaxis.labelpad = 2
        ax.yaxis.labelpad = 2
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Model folder containing step 12's structure_agreement/")
    parser.add_argument("--language", choices=tuple(LANGUAGES), required=True)
    parser.add_argument("--misuse", "--misuse-id", choices=MISUSES, required=True)
    parser.add_argument("--model-name", help="Display name; otherwise inferred from the model folder")
    parser.add_argument("--bins", type=int, default=80, help="Number of bins in each panel (default: 80)")
    parser.add_argument("--limit", type=int,
                        help="First N complete originals in the selected cell, ordered by task and sample")
    parser.add_argument("--output", type=Path, help="Output PDF path")
    args = parser.parse_args()
    if args.bins < 2:
        parser.error("--bins must be at least 2")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    output = args.output or (args.input_dir / "figures" /
                             f"structure_agreement_{args.language}_{args.misuse}.pdf")
    if output.suffix.lower() != ".pdf":
        parser.error("--output must end in .pdf")
    cosines, norms, layer = load_cell(args.input_dir, args.language, args.misuse, args.limit)
    model_name = args.model_name or MODEL_NAMES.get(args.input_dir.name, args.input_dir.name)
    title = f"{model_name} {LANGUAGES[args.language]} {args.misuse}, Layer {layer}"
    originals = len(norms)
    print(f"{title}: {originals} originals; each structure has "
          f"{3 * originals} cosines and {originals} raw norms; {args.bins} bins per panel.")
    fig = plot_agreement(cosines, norms, title, args.bins)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", facecolor="white")
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
