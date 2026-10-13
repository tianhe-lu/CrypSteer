"""Plot saved additive fits of repair combinations for one or more models."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from tqdm import tqdm


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ("java", "typescript", "csharp", "python")
MODEL_NAMES = {
    "llama": "Llama", "codegemma": "CodeGemma", "qwen": "Qwen",
    "qwen_coder": "Qwen-Coder", "deepseek": "DeepSeek",
}


def load_fit(model_dir, language):
    folder = model_dir / "combination_analysis"
    with (folder / "by_language.csv").open(encoding="utf-8", newline="") as handle:
        metrics = [row for row in csv.DictReader(handle) if row["language"] == language]
    if len(metrics) != 1 or int(metrics[0]["programs"]) != 1024:
        raise ValueError(f"{folder}: expected one complete 1,024-program fit for {language}")
    with (folder / "projections.csv").open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["language"] == language]
    if len(rows) != 1024 or {row["secure_bits"] for row in rows} != {f"{i:010b}" for i in range(1024)}:
        raise ValueError(f"{folder}: expected each ten-bit configuration exactly once for {language}")
    rows.sort(key=lambda row: row["secure_bits"], reverse=True)
    # These are the full-configuration OLS fit and endpoint-centered projection.
    # p_pred and p_from_misuse belong to the separate eleven-anchor analysis.
    fitted = np.asarray([float(row["p_fit"]) for row in rows], dtype=np.float64)
    observed = np.asarray([float(row["p_midpoint"]) for row in rows], dtype=np.float64)
    if not np.isfinite(fitted).all() or not np.isfinite(observed).all():
        raise ValueError(f"{folder}: expected finite fitted and measured projections")
    r2 = float(metrics[0]["fit_r2"]) if metrics[0]["fit_r2"] else None
    # Check the saved score against these points, without fitting again.
    total = float(np.sum((observed - observed.mean()) ** 2))
    if total == 0:
        if r2 is not None:
            raise ValueError(f"{folder}: a constant projection should have undefined R2")
    else:
        point_r2 = 1 - float(np.sum((observed - fitted) ** 2)) / total
        if r2 is None or not np.isfinite(r2) or not np.isclose(r2, point_r2, atol=1e-10, rtol=1e-8):
            raise ValueError(f"{folder}: saved fit R2 does not match the plotted points")
    return fitted, observed, r2


def plot_fits(results, model_names):
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7,
        "axes.linewidth": 0.5, "pdf.fonttype": 42,
    })
    # Use the paper's square panels and one-row layout; five models span 16 cm.
    width, height = max(3.77, 3.72 + 3.07 * (len(results) - 1)), 3.8
    side, spacing = 2.32, 3.07
    fig = plt.figure(figsize=(width / 2.54, height / 2.54))
    for index, ((fitted, observed, r2), name) in enumerate(tqdm(
            list(zip(results, model_names)), desc="Plot combination fits", unit="model")):
        left = 1.02 + index * spacing
        ax = fig.add_axes([left / width, 1.1 / height, side / width, side / height])
        lower = float(min(fitted.min(), observed.min()))
        upper = float(max(fitted.max(), observed.max()))
        padding = 0.045 * (upper - lower if upper > lower else max(abs(upper), 1.0))
        limits = (lower - padding, upper + padding)
        ax.plot(limits, limits, lw=0.65, ls=(0, (4, 3)), color="#606060", zorder=1)
        ax.scatter(fitted, observed, s=1.2, color="#28799e", alpha=0.43,
                   edgecolors="none", rasterized=False, zorder=2)
        ax.set(xlim=limits, ylim=limits)
        ax.set_box_aspect(1)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(labelsize=7, length=1.8, width=0.6, pad=1.5)
        ax.spines[["top", "right"]].set_visible(False)
        score = r"$R^2=\mathrm{undefined}$" if r2 is None else f"$R^2={r2:.3f}$"
        ax.text(0.04, 0.96, score, transform=ax.transAxes, va="top", fontsize=7)
        fig.text((left + side / 2) / width, 0.55 / height, name,
                 ha="center", va="center", fontsize=8)

    center = (1.02 + (len(results) - 1) * spacing / 2 + side / 2) / width
    # Keep these labels identical to the paper's additive-fit figure.
    fig.text(center, 0.018, "Additive prediction", ha="center", fontsize=8)
    fig.text(0.224 / width, 0.61, "Measured projection", va="center",
             ha="center", rotation=90, fontsize=8)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", "--input-dir", nargs="+", type=Path, required=True,
                        help="Model folders containing step 16's combination_analysis/; plotted in this order")
    parser.add_argument("--language", choices=LANGUAGES, default="java",
                        help="Language to plot (default: java)")
    parser.add_argument("--model-names", nargs="+", help="Optional display names, one per input folder")
    parser.add_argument("--output", type=Path, help="Output PDF path")
    parser.add_argument("--limit", type=int, help="Plot the first N input models, retaining all 1,024 combinations")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.model_names is not None and len(args.model_names) != len(args.input_dirs):
        parser.error("--model-names must provide one name per input folder")
    names = args.model_names or [MODEL_NAMES.get(folder.name, folder.name) for folder in args.input_dirs]
    folders, names = args.input_dirs[:args.limit], names[:args.limit]
    output = args.output
    if output is None:
        parent = folders[0] if len(folders) == 1 else ARTIFACT_ROOT / "outputs"
        output = parent / "figures" / f"combination_fit_{args.language}.pdf"
    if output.suffix.lower() != ".pdf":
        parser.error("--output must end in .pdf")
    results = [load_fit(folder, args.language) for folder in
               tqdm(folders, desc="Read combination fits", unit="model")]
    for (fitted, _, r2), name in zip(results, names):
        score = "undefined" if r2 is None else f"{r2:.6f}"
        print(f"{name} / {args.language}: {len(fitted)} combinations; saved fit R2={score}")
    fig = plot_fits(results, names)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", facecolor="white")
    plt.close(fig)
    print(f"Saved {len(results)} model(s): {output}")


if __name__ == "__main__":
    main()
