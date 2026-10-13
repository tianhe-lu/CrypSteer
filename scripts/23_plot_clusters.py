"""Plot metric-MDS views of cell directions with saved full-dimensional clusters."""

import argparse
import csv
import inspect
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import numpy as np
import sklearn
from sklearn.manifold import MDS
from tqdm import tqdm


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ("java", "typescript", "csharp", "python")
LANGUAGE_NAMES = ("Java", "TypeScript", "C#", "Python")
MISUSES = tuple(f"M{i:02d}" for i in range(1, 11))
MARKERS = ("o", "^", "s", "D")
MODEL_NAMES = {
    "llama": "Llama", "codegemma": "CodeGemma", "qwen": "Qwen",
    "qwen_coder": "Qwen-Coder", "deepseek": "DeepSeek",
}


def load_clusters(model_dir):
    folder = model_dir / "clustering"
    with (folder / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    with (folder / "assignments.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    path = folder / "cell_directions.npy"
    if not path.is_file():
        raise FileNotFoundError(f"{path}: run the updated step 14 to save cell directions")
    directions = np.load(path, allow_pickle=False)
    saved_cells = [(language, misuse) for language in LANGUAGES for misuse in MISUSES]
    if [(row["language"], row["misuse_id"]) for row in rows] != saved_cells:
        raise ValueError(f"{folder}: expected step 14's 40 rows in language/category order")
    if (directions.shape != (40, summary["hidden_size"]) or directions.dtype != np.float32
            or not np.isfinite(directions).all()):
        raise ValueError(f"{folder}: expected 40 finite float32 cell directions")
    if not np.allclose(np.linalg.norm(directions.astype(np.float64), axis=1), 1, atol=2e-6, rtol=0):
        raise ValueError(f"{folder}: expected unit directions")
    clusters = np.asarray([int(row["cluster"]) for row in rows], dtype=int)
    if summary["config"]["cluster_count"] != 10 or set(clusters.tolist()) != set(range(10)):
        raise ValueError(f"{folder}: expected ten saved clusters numbered 0–9")
    # Match the old MDS input order, which also determines random initialization:
    # M01 across four languages, then M02, etc. Reorder vectors and labels together.
    order = [saved_cells.index((language, misuse)) for misuse in MISUSES for language in LANGUAGES]
    return directions[order].astype(np.float64), [rows[i] for i in order], clusters[order], summary


def project_mds(directions):
    # A PCA reference fixes only the orientation of the final two-dimensional view.
    centered = directions - directions.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    if singular[1] <= 0:
        raise ValueError("The cell directions must span a two-dimensional centered display")
    basis = vt[:2].T.copy()
    for axis in range(2):
        if basis[np.argmax(np.abs(basis[:, axis])), axis] < 0:
            basis[:, axis] *= -1
    pca = centered @ basis
    distances = np.linalg.norm(directions[:, None, :] - directions[None, :, :], axis=2)
    options = dict(n_components=2, n_init=8, max_iter=1000, eps=1e-9,
                   random_state=0, n_jobs=1, normalized_stress=False)
    # scikit-learn 1.8 renamed the two metric arguments and added constructor init.
    if "metric_mds" in inspect.signature(MDS).parameters:
        options.update(metric_mds=True, metric="precomputed", init="random")
    else:
        options.update(metric=True, dissimilarity="precomputed")
    fitted = MDS(**options)
    coordinates = fitted.fit_transform(distances)
    coordinates -= coordinates.mean(axis=0, keepdims=True)
    left, _, right = np.linalg.svd(coordinates.T @ pca)
    coordinates = coordinates @ (left @ right)
    # Report distortion in the console; it is not a clustering score.
    displayed = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)
    upper = np.triu_indices(40, k=1)
    error = float(np.sqrt(np.sum((distances[upper] - displayed[upper]) ** 2)
                          / np.sum(distances[upper] ** 2)))
    return coordinates, error, int(fitted.n_iter_)


def overlap_area(first, second):
    width = max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0))
    height = max(0.0, min(first.y1, second.y1) - max(first.y0, second.y0))
    return width * height


def label_points(fig, ax, coordinates, rows):
    # Move labels alone to reduce overlap; the MDS points stay in place.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pixels_per_point = fig.dpi / 72
    points = ax.transData.transform(coordinates)
    radius = 3 * pixels_per_point
    marker_boxes = [Bbox.from_bounds(x - radius, y - radius, 2 * radius, 2 * radius)
                    for x, y in points]
    font = FontProperties(family="STIXGeneral", size=5.5)
    candidates = [(r * np.cos(angle), r * np.sin(angle))
                  for r in (7, 11, 16, 22, 29, 37)
                  for angle in np.deg2rad((45, 135, 225, 315, 0, 90, 180, 270))]
    bounds, placed = ax.get_window_extent(renderer), []
    for row, point, xy in zip(rows, points, coordinates):
        label = row["misuse_id"]
        width, height, _ = renderer.get_text_width_height_descent(label, font, ismath=False)
        width, height = width + 2, height + 2
        choices = []
        for dx, dy in candidates:
            x, y = point + np.asarray((dx, dy)) * pixels_per_point
            box = Bbox.from_bounds(x - width / 2, y - height / 2, width, height)
            outside = width * height - overlap_area(box, bounds)
            overlap = sum(overlap_area(box, other) for other in placed)
            marker_overlap = sum(overlap_area(box, marker) for marker in marker_boxes)
            score = 1000 * outside + 100 * overlap + 40 * marker_overlap + dx * dx + dy * dy
            choices.append((score, dx, dy, box))
        _, dx, dy, box = min(choices, key=lambda item: item[0])
        placed.append(box)
        ax.annotate(label, xy=xy, xytext=(dx, dy), textcoords="offset points",
                    ha="center", va="center", fontsize=5.5, color="#222222", zorder=4,
                    arrowprops={"arrowstyle": "-", "color": "#999999", "lw": 0.3,
                                "shrinkA": 1, "shrinkB": 3},
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.1})


def plot_clusters(results, model_names):
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
        "axes.linewidth": 0.5, "pdf.fonttype": 42,
    })
    columns = min(2, len(results))
    row_count = (len(results) + columns - 1) // columns
    width, height = 7.7 * columns, 6.2 * row_count
    fig = plt.figure(figsize=(width / 2.54, height / 2.54))
    palette = plt.get_cmap("tab10").colors
    for index, ((coordinates, rows, clusters, summary), name) in enumerate(tqdm(
            list(zip(results, model_names)), desc="Plot clusters", unit="model")):
        left = (index % columns) * 7.7
        bottom = height - (index // columns + 1) * 6.2
        ax = fig.add_axes([(left + 1.309) / width, (bottom + 1.55) / height,
                           6.237 / width, 4.092 / height])
        for language, marker in zip(LANGUAGES, MARKERS):
            indices = [i for i, row in enumerate(rows) if row["language"] == language]
            ax.scatter(coordinates[indices, 0], coordinates[indices, 1], s=19,
                       c=[palette[clusters[i]] for i in indices], marker=marker,
                       edgecolors="#333333", linewidths=0.3, zorder=3, rasterized=False)
        ax.set(xlabel="MDS 1", ylabel="MDS 2", aspect="equal")
        ax.margins(x=0.18, y=0.18)
        ax.tick_params(length=2, pad=2, labelsize=6)
        ax.xaxis.label.set_size(7)
        ax.yaxis.label.set_size(7)
        ax.xaxis.labelpad = ax.yaxis.labelpad = 2
        ax.spines[["top", "right"]].set_visible(False)
        title = f"{name}, Layer {summary['layer']}"
        if not summary["full_source"]:
            title += " (subset)"
        fig.text((left + 3.85) / width, (bottom + 5.95) / height, title,
                 ha="center", va="center", fontsize=8)
        label_points(fig, ax, coordinates, rows)
    handles = [Line2D([], [], linestyle="none", marker=marker, markersize=4,
                      markerfacecolor="#cccccc", markeredgecolor="#333333", markeredgewidth=0.3,
                      label=name) for marker, name in zip(MARKERS, LANGUAGE_NAMES)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.093 / height),
               ncol=4, frameon=False, fontsize=6.5, handletextpad=0.3, columnspacing=0.8)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", "--input-dir", nargs="+", type=Path, required=True,
                        help="Model folders containing step 14's clustering/ with cell_directions.npy")
    parser.add_argument("--model-names", "--model-name", nargs="+", help="Display names, one per input folder")
    parser.add_argument("--output", type=Path, help="Output PDF path")
    parser.add_argument("--limit", type=int, help="Plot the first N input models, keeping all 40 cells")
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
        output = parent / "figures" / "clusters.pdf"
    if output.suffix.lower() != ".pdf":
        parser.error("--output must end in .pdf")
    print(f"Metric MDS: scikit-learn {sklearn.__version__}, seed 0, 8 starts, max_iter=1000, eps=1e-9.")
    results = []
    for folder, name in tqdm(list(zip(folders, names)), desc="Project cell directions", unit="model"):
        directions, rows, clusters, summary = load_clusters(folder)
        coordinates, error, iterations = project_mds(directions)
        results.append((coordinates, rows, clusters, summary))
        print(f"{name}: 40 cells; MDS iterations={iterations}, distance error={error:.6f}; "
              f"saved category ARI={summary['category_ari']:.6f}")
    fig = plot_clusters(results, names)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Subject": f"Metric MDS; scikit-learn {sklearn.__version__}; seed=0; n_init=8; "
                "max_iter=1000; eps=1e-9; Euclidean distances; rigid alignment to PCA; clusters from step 14."}
    fig.savefig(output, format="pdf", facecolor="white", metadata=metadata)
    plt.close(fig)
    print(f"Saved {len(results)} model(s): {output}")


if __name__ == "__main__":
    main()
