"""Plot saved cell cosines and repair norm ratios for one or more models."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
from tqdm import tqdm


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ("java", "typescript", "csharp", "python")
LANGUAGE_NAMES = ("Java", "TypeScript", "C#", "Python")
LANGUAGE_COLORS = ("#0072B2", "#E69F00", "#7A5195", "#CC79A7")
MISUSES = tuple(f"M{i:02d}" for i in range(1, 11))
MODEL_NAMES = {
    "llama": "Llama", "codegemma": "CodeGemma", "qwen": "Qwen",
    "qwen_coder": "Qwen-Coder", "deepseek": "DeepSeek",
}


def load_geometry(model_dir):
    folder = model_dir / "geometry"
    with (folder / "cells.csv").open(encoding="utf-8", newline="") as handle:
        cells = list(csv.DictReader(handle))
    expected = [(language, misuse) for misuse in MISUSES for language in LANGUAGES]
    rows = {(cell["language"], cell["misuse_id"]): int(cell["row"]) for cell in cells}
    if (len(cells) != 40 or set(rows) != set(expected)
            or sorted(rows.values()) != list(range(40))):
        raise ValueError(f"{folder}: expected all 40 language/category cells and rows 0–39")
    order = [rows[cell] for cell in expected]
    with np.load(folder / "geometry.npz", allow_pickle=False) as data:
        cosine = data["cosine"]
        norm_ratio = data["norm_ratio"]
    for name, matrix in (("cosine", cosine), ("norm_ratio", norm_ratio)):
        if matrix.shape != (40, 40) or not np.isfinite(matrix).all():
            raise ValueError(f"{folder}: {name} must be a finite 40 × 40 matrix")
    if np.any(np.abs(cosine) > 1):
        raise ValueError(f"{folder}: cosine values must lie in [-1, 1]")
    if np.any(norm_ratio <= 0):
        raise ValueError(f"{folder}: norm ratios must be positive")
    log_ratio = np.log2(norm_ratio)
    if np.max(np.abs(log_ratio)) > 5 + 1e-10:
        raise ValueError(f"{folder}: norm ratios exceed the paper's color range [1/32, 32]")
    with (folder / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    # Reorder both axes together. Ratios remain row norm / column norm.
    return cosine[np.ix_(order, order)], log_ratio[np.ix_(order, order)], summary


def draw_matrix(fig, rectangle, values, cmap, bound, show_x_labels):
    ax = fig.add_axes(rectangle)
    edges = np.arange(41)
    mesh = ax.pcolormesh(
        edges, edges, values, cmap=cmap, vmin=-bound, vmax=bound,
        shading="flat", rasterized=False, antialiased=False,
        edgecolors="face", linewidth=0.1,
    )
    ax.set(xlim=(0, 40), ylim=(40, 0), aspect="equal")
    centers = np.arange(2, 40, 4)
    ax.set_xticks(centers)
    ax.set_yticks(centers, labels=MISUSES)
    ax.set_xticklabels(MISUSES if show_x_labels else [""] * 10, rotation=90)
    ax.tick_params(axis="both", labelsize=5.5, length=0, pad=2)
    for boundary in range(4, 40, 4):
        ax.axhline(boundary, color="white", linewidth=0.25)
        ax.axvline(boundary, color="white", linewidth=0.25)
    return mesh


def plot_geometry(model_dirs, model_names):
    plt.rcParams.update({
        "font.family": "STIXGeneral", "font.size": 7,
        "mathtext.fontset": "stix", "pdf.fonttype": 42,
        "axes.linewidth": 0.5,
    })
    cosine_cmap = LinearSegmentedColormap.from_list(
        "repair_cosine", ("#c57435", "#faf9f6", "#2c6a9f"), N=257,
    )
    ratio_cmap = LinearSegmentedColormap.from_list(
        "repair_ratio", ("#2c6a9f", "#faf9f6", "#c57435"), N=257,
    )
    language_cmap = ListedColormap(LANGUAGE_COLORS)
    # Dimensions in cm; matrix size and text size stay fixed as rows are added.
    width, side, row_spacing = 8.4, 3.1, 3.7
    height = 5.5 + (len(model_dirs) - 1) * row_spacing
    lefts = (0.64, 4.64)
    fig = plt.figure(figsize=(width / 2.54, height / 2.54))

    def rectangle(left, bottom, w, h):
        return [left / width, bottom / height, w / width, h / height]

    for left, title in zip(lefts, ("Cosine similarity", "Norm ratio")):
        fig.text((left + side / 2) / width, (height - 0.12) / height,
                 title, ha="center", va="top", fontsize=7)

    meshes = []
    for row, (folder, name) in enumerate(tqdm(
            list(zip(model_dirs, model_names)), desc="Plot geometry", unit="model")):
        cosine, log_ratio, summary = load_geometry(folder)
        top = height - 0.85 - row * row_spacing
        bottom = top - side
        title = f"{name} · Layer {summary['layer']}"
        if not summary["full_source"]:
            title += " (subset)"
        fig.text(0.5, (top + 0.15) / height, title,
                 ha="center", va="bottom", fontsize=7, fontweight="bold")
        meshes = []
        for left, values, cmap, bound in zip(
                lefts, (cosine, log_ratio), (cosine_cmap, ratio_cmap), (1, 5)):
            meshes.append(draw_matrix(
                fig, rectangle(left, bottom, side, side), values, cmap, bound,
                show_x_labels=row == len(model_dirs) - 1,
            ))
            # Four language colors repeat within every category on both axes.
            languages = np.tile(np.arange(4), 10)
            for rect, colors in (
                (rectangle(left, top + 0.035, side, 0.065), languages[None, :]),
                (rectangle(left + side + 0.035, bottom, 0.065, side), languages[:, None]),
            ):
                strip = fig.add_axes(rect)
                strip.imshow(colors, cmap=language_cmap, vmin=-0.5, vmax=3.5,
                             aspect="auto", interpolation="nearest", origin="upper")
                strip.set_axis_off()

    # All rows use the same two scales, with one colorbar per column.
    for left, mesh, ticks, labels in zip(
            lefts, meshes, ((-1, 0, 1), (-5, -2, 0, 2, 5)),
            (("−1", "0", "1"), ("1/32", "1/4", "1", "4", "32"))):
        bar_ax = fig.add_axes(rectangle(left, 0.72, side, 0.11))
        bar = fig.colorbar(mesh, cax=bar_ax, orientation="horizontal", ticks=ticks)
        bar.ax.set_xticklabels(labels)
        bar.ax.tick_params(labelsize=6, length=2, width=0.5, pad=1)
        bar.outline.set_linewidth(0.5)
    fig.legend(
        handles=[Patch(facecolor=color, label=name)
                 for name, color in zip(LANGUAGE_NAMES, LANGUAGE_COLORS)],
        loc="center", bbox_to_anchor=(0.5, 0.15 / height),
        ncol=4, frameon=False, fontsize=6, handlelength=0.9,
        handleheight=0.7, columnspacing=1.0, handletextpad=0.5, borderaxespad=0,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", "--input-dir", nargs="+", type=Path, required=True,
                        help="Model folders containing step 06's geometry/; plotted in this order")
    parser.add_argument("--model-names", nargs="+",
                        help="Optional display names, one per input folder")
    parser.add_argument("--output", type=Path, help="Output PDF path")
    parser.add_argument("--limit", type=int,
                        help="Plot only the first N input models, retaining all 40 cells per model")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.model_names is not None and len(args.model_names) != len(args.input_dirs):
        parser.error("--model-names must provide one name per input folder")
    names = args.model_names or [MODEL_NAMES.get(p.name, p.name) for p in args.input_dirs]
    folders, names = args.input_dirs[:args.limit], names[:args.limit]
    output = args.output
    if output is None:
        parent = folders[0] if len(folders) == 1 else ARTIFACT_ROOT / "outputs"
        output = parent / "figures" / "geometry.pdf"
    if output.suffix.lower() != ".pdf":
        parser.error("--output must end in .pdf")
    fig = plot_geometry(folders, names)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", facecolor="white")
    plt.close(fig)
    print(f"Saved {len(folders)} model(s): {output}")


if __name__ == "__main__":
    main()
