"""Plot CrypSteer strength responses from saved step-09 summaries as a PDF."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
STRENGTHS = (-1.0, 0.0, 0.5, 1.0, 2.0)
METRICS = ("secure@1", "valid@1", "secure-valid@1")
MODEL_STYLES = {
    "llama": ("Llama", "#0072B2", "o"),
    "codegemma": ("CodeGemma", "#D55E00", "s"),
    "qwen": ("Qwen", "#009E73", "^"),
    "qwen_coder": ("Qwen-Coder", "#CC79A7", "D"),
    "deepseek": ("DeepSeek", "#E69F00", "v"),
}


def load_response(model_dir):
    path = model_dir / "summary" / "overall.csv"
    points = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["source_scope"] != "full" or row["method"] not in ("baseline", "crypsteer"):
                continue
            alpha = float(row["alpha"])
            # Zero uses unsteered generation, not a zero-strength intervention.
            if not ((row["method"] == "baseline" and alpha == 0)
                    or (row["method"] == "crypsteer" and alpha in STRENGTHS and alpha != 0)):
                continue
            if alpha in points:
                raise ValueError(f"{path}: duplicate full-source result at strength {alpha}")
            if (row["complete"].lower() != "true" or int(row["n"]) <= 0
                    or int(row["n"]) != int(row["expected_n"])
                    or int(row["cells"]) <= 0 or int(row["cells"]) != int(row["expected_cells"])):
                raise ValueError(f"{path}: incomplete result for {row['setting']}")
            for metric in METRICS:
                rate = float(row[metric])
                if not math.isfinite(rate) or not 0 <= rate <= 100:
                    raise ValueError(f"{path}: invalid {metric} for {row['setting']}")
            points[alpha] = row

    missing = sorted(set(STRENGTHS) - points.keys())
    if missing:
        raise ValueError(f"{path}: missing full-source strengths {missing}")
    rows = [points[alpha] for alpha in STRENGTHS]
    models = {row["model"] for row in rows}
    if len(models) != 1:
        raise ValueError(f"{path}: expected results for one model")
    coverage = {(int(row["n"]), int(row["cells"])) for row in rows}
    if len(coverage) != 1:
        raise ValueError(f"{path}: strengths have different sample or cell counts")
    settings = {(int(points[alpha]["layer"]), float(points[alpha]["sigma"]))
                for alpha in STRENGTHS if alpha != 0}
    if len(settings) != 1:
        raise ValueError(f"{path}: CrypSteer strengths use different layers or scales")
    _, sigma = next(iter(settings))
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"{path}: expected a positive, finite intervention scale")
    return {
        "model": rows[0]["model"],
        # Step 09 already stores percentages. Preserve its equal-cell averages.
        "rates": {metric: [float(row[metric]) for row in rows] for metric in METRICS},
    }


def plot_responses(responses, model_names):
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7.2,
        "axes.titlesize": 7.4, "axes.titleweight": "bold", "axes.labelsize": 7.4,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42,
    })
    # The paper uses one column with three stacked metrics and a shared legend.
    legend_rows = math.ceil(len(responses) / 3)
    height_cm = 8.0 + 0.45 * max(0, legend_rows - 2)
    top = 1 - (0.75 + 0.3 * legend_rows) / height_cm
    fig, axes = plt.subplots(3, 1, figsize=(7.7 / 2.54, height_cm / 2.54),
                             sharex=True, sharey=True)
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.13, top=top, hspace=0.4)
    for panel, (ax, metric) in enumerate(zip(axes, METRICS)):
        ax.set_title(f"({chr(ord('a') + panel)}) {metric}", loc="left", pad=3)
        ax.set_xlim(-1.15, 2.15)
        ax.set_ylim(0, 100)
        ax.set_xticks(STRENGTHS, ("−1", "0", "0.5", "1", "2"))
        ax.set_yticks((0, 50, 100))
        ax.set_yticks((25, 75), minor=True)
        ax.set_axisbelow(True)
        ax.grid(axis="y", which="both", color="#E1E1E1", linewidth=0.4)
        ax.tick_params(axis="y", which="minor", length=0)
        ax.axvline(0, color="#B8B8B8", linewidth=0.5, linestyle=(0, (2, 2)), zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        for index, response in enumerate(responses):
            model = response["model"]
            name, color, marker = MODEL_STYLES.get(model, (model, f"C{index % 10}", "o"))
            if model_names is not None:
                name = model_names[index]
            # Straight segments connect the five observed rates; no curve fitting.
            ax.plot(STRENGTHS, response["rates"][metric], label=name,
                    color=color, marker=marker, linewidth=0.9, markersize=2.8,
                    markeredgewidth=0.35, markeredgecolor="white", zorder=3)
    fig.supylabel("Rate (%)", fontsize=7.4, x=0.025, y=0.48)
    axes[-1].set_xlabel("Steering strength (model-specific σ)", labelpad=4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.995),
               ncol=min(3, len(responses)), frameon=False, fontsize=7.2,
               handlelength=1.5, handletextpad=0.4, columnspacing=1.0, labelspacing=0.5)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", "--input-dir", nargs="+", type=Path, required=True,
                        help="Model folders containing step 09's summary/overall.csv")
    parser.add_argument("--model-names", nargs="+",
                        help="Optional display names, one per input folder")
    parser.add_argument("--output", type=Path, help="Output PDF path")
    parser.add_argument("--limit", type=int, help="Plot only the first N input models")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.model_names is not None and len(args.model_names) != len(args.input_dirs):
        parser.error("--model-names must provide one name per input folder")
    folders = args.input_dirs[:args.limit]
    names = args.model_names[:args.limit] if args.model_names is not None else None
    output = args.output
    if output is None:
        parent = folders[0] if len(folders) == 1 else ARTIFACT_ROOT / "outputs"
        output = parent / "figures" / "strength_response.pdf"
    if output.suffix.lower() != ".pdf":
        parser.error("--output must end in .pdf")
    responses = [load_response(folder) for folder in
                 tqdm(folders, desc="Read strength responses", unit="model")]
    fig = plot_responses(responses, names)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", facecolor="white")
    plt.close(fig)
    print(f"Saved {len(responses)} model(s), {len(responses) * 15} observed rates: {output}")


if __name__ == "__main__":
    main()
