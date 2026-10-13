"""Plot saved cross-layer repair-geometry similarities for one or more models."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from tqdm import tqdm


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = {
    "llama": "Llama", "codegemma": "CodeGemma", "qwen": "Qwen",
    "qwen_coder": "Qwen-Coder", "deepseek": "DeepSeek",
}


def load_rsa(model_dir):
    folder = model_dir / "layer_rsa"
    with (folder / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    layers = summary["layers"]
    if (not layers or any(type(layer) is not int or layer < 0 for layer in layers)
            or layers != sorted(set(layers))):
        raise ValueError(f"{folder}: expected distinct, increasing layer indices")
    if summary["similarity"] != "cosine":
        raise ValueError(f"{folder}: expected cosine between layer-geometry signatures")
    labels = [f"L{layer}" for layer in layers]
    with (folder / "cross_layer_rsa.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    if (header != ["label", *labels] or len(rows) != len(labels)
            or any(len(row) != len(labels) + 1 for row in rows)
            or [row[0] for row in rows] != labels):
        raise ValueError(f"{folder}: matrix rows and columns must match summary layers")
    matrix = np.asarray([[float(value) for value in row[1:]] for row in rows], dtype=np.float64)
    finite = matrix[np.isfinite(matrix)]
    # Step 15 preserves float32 roundoff, including diagonal entries just above 1.
    if np.isinf(matrix).any() or np.any(np.abs(finite) > 1 + 1e-6):
        raise ValueError(f"{folder}: expected cosines in [-1, 1], allowing float32 roundoff")
    if not np.allclose(matrix, matrix.T, atol=1e-6, rtol=0, equal_nan=True):
        raise ValueError(f"{folder}: expected a symmetric cross-layer matrix")
    return matrix, labels, summary


def plot_rsa(results, model_names):
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7,
        "axes.linewidth": 0.5, "pdf.fonttype": 42,
    })
    cmap = LinearSegmentedColormap.from_list(
        "rsa_blue_green_yellow", plt.get_cmap("viridis")(np.linspace(0.28, 0.97, 256)),
    )
    cmap.set_bad("#d9d9d9")
    # The paper's matrices use 0–1. Extend the shared scale for signed results.
    lower = -1 if any(np.any(matrix < 0) for matrix, _, _ in results) else 0
    ticks = (-1, 0, 1) if lower < 0 else (0, 0.5, 1)
    tick_labels = ("−1", "0", "1") if lower < 0 else ("0", "0.5", "1")

    # Keep the paper's panel size; five models occupy a 16 × 4.2 cm row.
    width, height, side, spacing = max(3.77, 3.48 + 3.13 * (len(results) - 1)), 4.2, 2.45, 3.13
    fig = plt.figure(figsize=(width / 2.54, height / 2.54))
    for index, ((matrix, labels, summary), name) in enumerate(tqdm(
            list(zip(results, model_names)), desc="Plot layer RSA", unit="model")):
        left = 0.65 + index * spacing
        ax = fig.add_axes([left / width, 1.35 / height, side / width, side / height])
        count = len(labels)
        edges = np.arange(count + 1)
        # Use the saved signed values directly; NaNs stay masked rather than zero.
        mesh = ax.pcolormesh(
            edges, edges, np.ma.masked_invalid(matrix), cmap=cmap, vmin=lower, vmax=1,
            shading="flat", rasterized=False, antialiased=False,
            edgecolors="face", linewidth=0.6,
        )
        ax.set(xlim=(0, count), ylim=(count, 0), aspect="equal")
        positions = np.linspace(0, count - 1, min(3, count)).astype(int)
        layer_labels = [labels[i] for i in positions]
        ax.set_xticks(positions + 0.5, labels=layer_labels)
        ax.set_yticks(positions + 0.5, labels=layer_labels)
        ax.tick_params(length=2, width=0.5, pad=2, labelsize=7)
        title = name if summary["full_source"] else f"{name} (subset)"
        fig.text((left + side / 2) / width, 0.72 / height, title,
                 ha="center", va="center", fontsize=8)

    bar_width = min(5.8, width - 1.2)
    bar_ax = fig.add_axes([(width - bar_width) / (2 * width), 0.36 / height,
                           bar_width / width, 0.15 / height])
    bar = fig.colorbar(mesh, cax=bar_ax, orientation="horizontal", ticks=ticks)
    bar.set_ticklabels(tick_labels)
    bar.ax.tick_params(labelsize=6.5, length=2, width=0.5, pad=2)
    bar.outline.set_linewidth(0.5)
    bar.solids.set_rasterized(False)
    bar.solids.set_edgecolor("face")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", "--input-dir", nargs="+", type=Path, required=True,
                        help="Model folders containing step 15's layer_rsa/; plotted in this order")
    parser.add_argument("--model-names", nargs="+", help="Optional display names, one per input folder")
    parser.add_argument("--output", type=Path, help="Output PDF path")
    parser.add_argument("--limit", type=int, help="Plot the first N input models, retaining all saved layers")
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
        output = parent / "figures" / "layer_rsa.pdf"
    if output.suffix.lower() != ".pdf":
        parser.error("--output must end in .pdf")
    results = [load_rsa(folder) for folder in
               tqdm(folders, desc="Read layer RSA", unit="model")]
    for (matrix, _, _), name in zip(results, names):
        print(f"{name}: {len(matrix)} × {len(matrix)} saved similarities; "
              f"{np.isnan(matrix).sum()} undefined entries (gray).")
    fig = plot_rsa(results, names)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", facecolor="white")
    plt.close(fig)
    print(f"Saved {len(results)} model(s): {output}")


if __name__ == "__main__":
    main()
