from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.reporting.pareto import KNOWN_DATASETS, infer_dataset
from src.utils.paths import ensure_dir


def _group_by_dataset(rows: list[dict[str, str]]) -> list[tuple[str, list[dict[str, str]]]]:
    """Partition rows by dataset (canonical order) so each gets its own figure; mAP and
    latency are not comparable across datasets, so pooling them on one axis is misleading."""
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(infer_dataset(row), []).append(row)
    ordered = [d for d in KNOWN_DATASETS if d in groups] + [d for d in groups if d not in KNOWN_DATASETS]
    return [(d, groups[d]) for d in ordered]


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except ValueError:
                continue
    return None


def _save(fig: plt.Figure, output_root: Path, paper_root: Path, name: str) -> None:
    for root in (output_root, paper_root):
        ensure_dir(root)
        fig.savefig(root / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(root / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _bar(
    rows: list[dict[str, str]],
    metric: str,
    title: str,
    output_root: Path,
    paper_root: Path,
    name: str | None = None,
) -> None:
    values = [(row.get("run_name", ""), _float(row, metric)) for row in rows]
    values = [(name_, value) for name_, value in values if value is not None]
    if not values:
        return
    names, ys = zip(*values)
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 0.7), 4.5))
    ax.bar(names, ys, color="#356A75")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output_root, paper_root, name or (metric.replace("_", "-") + "-comparison"))


def _scatter(
    rows: list[dict[str, str]],
    x_key: str,
    y_key: str,
    name: str,
    xlabel: str,
    ylabel: str,
    output_root: Path,
    paper_root: Path,
    highlight_key: str | None = None,
) -> None:
    points = [
        (row.get("run_name", ""), _float(row, x_key), _float(row, y_key))
        for row in rows
    ]
    points = [(n, x, y) for n, x, y in points if x is not None and y is not None]
    if not points:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    row_by_name = {row.get("run_name", ""): row for row in rows}
    for label, x, y in points:
        highlighted = highlight_key and str(row_by_name.get(label, {}).get(highlight_key)).lower() == "true"
        ax.scatter(
            x,
            y,
            s=80 if highlighted else 50,
            color="#C44900" if highlighted else "#4B7F52",
            marker="D" if highlighted else "o",
        )
        ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    _save(fig, output_root, paper_root, name)


def _heatmap(rows: list[dict[str, str]], output_root: Path, paper_root: Path, name: str = "per-class-ap-heatmap") -> None:
    if not rows:
        return
    class_names = sorted({row.get("class_name", "") for row in rows if row.get("class_name")})
    run_names = sorted({row.get("run_name", "") for row in rows if row.get("run_name")})
    if not class_names or not run_names:
        return
    matrix = np.full((len(class_names), len(run_names)), np.nan)
    class_index = {name: idx for idx, name in enumerate(class_names)}
    run_index = {name: idx for idx, name in enumerate(run_names)}
    for row in rows:
        value = _float(row, "ap50_95", "ap50")
        if value is not None:
            matrix[class_index[row["class_name"]], run_index[row["run_name"]]] = value
    fig, ax = plt.subplots(figsize=(max(7, len(run_names) * 0.8), max(5, len(class_names) * 0.3)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(run_names)), run_names, rotation=35, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_title("Per-class AP heatmap")
    fig.colorbar(im, ax=ax, label="AP")
    _save(fig, output_root, paper_root, name)


def generate_figures(
    tables_dir: str | Path = "results/tables",
    output_dir: str | Path = "results/figures",
    paper_dir: str | Path = "paper_assets/figures",
) -> None:
    tables_root = Path(tables_dir)
    output_root = ensure_dir(output_dir)
    paper_root = ensure_dir(paper_dir)
    accuracy = _read_rows(tables_root / "main_accuracy_results.csv")
    tradeoff = _read_rows(tables_root / "accuracy_efficiency_tradeoff.csv")
    pareto = _read_rows(tables_root / "pareto_frontier.csv")
    quantization = _read_rows(tables_root / "quantization_pareto.csv")
    per_class = _read_rows(tables_root / "per_class_ap.csv")

    # One figure per dataset: mAP / latency are not comparable across datasets.
    for dataset, rows in _group_by_dataset(accuracy):
        suffix = f"-{dataset}" if dataset else ""
        _bar(rows, "map50_95", f"mAP@50:95 ({dataset or 'all'})", output_root, paper_root, name=f"map50-95-comparison{suffix}")
        _bar(rows, "map50", f"mAP@50 ({dataset or 'all'})", output_root, paper_root, name=f"map50-comparison{suffix}")
    for dataset, rows in _group_by_dataset(tradeoff):
        suffix = f"-{dataset}" if dataset else ""
        _scatter(rows, "fps", "map50_95", f"fps-vs-map{suffix}", "FPS", "mAP@50:95", output_root, paper_root)
        _scatter(rows, "parameters", "map50_95", f"parameters-vs-map{suffix}", "Parameters", "mAP@50:95", output_root, paper_root)
    for dataset, rows in _group_by_dataset(pareto):
        suffix = f"-{dataset}" if dataset else ""
        _scatter(rows, "fps", "map50_95", f"pareto-fps-vs-map{suffix}", "FPS", "mAP@50:95", output_root, paper_root, highlight_key="pareto_efficient")
        _scatter(rows, "latency_mean_ms", "map50_95", f"pareto-latency-vs-map{suffix}", "Latency (ms)", "mAP@50:95", output_root, paper_root, highlight_key="pareto_efficient")
    for dataset, rows in _group_by_dataset(quantization):
        suffix = f"-{dataset}" if dataset else ""
        _scatter(rows, "latency_mean_ms", "map50_95", f"quantization-latency-vs-map{suffix}", "Latency (ms/image)", "mAP@50:95", output_root, paper_root, highlight_key="pareto_efficient")
    for dataset, rows in _group_by_dataset(per_class):
        suffix = f"-{dataset}" if dataset else ""
        _heatmap(rows, output_root, paper_root, name=f"per-class-ap-heatmap{suffix}")
