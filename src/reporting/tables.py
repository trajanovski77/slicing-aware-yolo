from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.reporting.latex import csv_to_latex
from src.reporting.pareto import KNOWN_DATASETS, analyze_pareto, infer_dataset
from src.utils.paths import ensure_dir


def _read_csvs(root: Path, pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob(pattern)):
        if path.stat().st_size == 0:
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "run_name",
        "task",
        "split",
        "map50_95",
        "map50",
        "precision",
        "recall",
        "f1",
        "fps",
        "latency_mean_ms",
        "parameters",
        "model_size_mb",
    ]
    ordered = [c for c in preferred if c in columns] + [c for c in columns if c not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def generate_tables(
    metrics_dir: str | Path = "results/metrics",
    output_dir: str | Path = "results/tables",
    paper_dir: str | Path = "paper_assets/tables",
) -> dict[str, str]:
    metrics_root = Path(metrics_dir)
    out_root = ensure_dir(output_dir)
    paper_root = ensure_dir(paper_dir)
    accuracy = _read_csvs(metrics_root, "metrics.csv")
    per_class = _read_csvs(metrics_root, "per_class_ap.csv")
    speed_rows = _read_csvs(metrics_root / "speed", "*.csv") if (metrics_root / "speed").exists() else []
    ensemble_rows = _read_csvs(Path("results/predictions/ensembles"), "manifest.csv")

    for row in accuracy:
        row["dataset"] = infer_dataset(row)
    for row in per_class:
        row["dataset"] = infer_dataset(row)

    speed_by_run = {row.get("run_name"): row for row in speed_rows}
    tradeoff = []
    for row in accuracy:
        merged = dict(row)
        speed = speed_by_run.get(row.get("run_name"), {})
        for key in ("fps", "latency_mean_ms", "parameters", "flops", "model_size_mb"):
            if key in speed:
                merged[key] = speed[key]
        tradeoff.append(merged)

    outputs = {
        "accuracy": out_root / "main_accuracy_results.csv",
        "per_class": out_root / "per_class_ap.csv",
        "speed": out_root / "speed_efficiency_results.csv",
        "tradeoff": out_root / "accuracy_efficiency_tradeoff.csv",
        "ensemble": out_root / "ensemble_predictions.csv",
    }
    _write_csv(accuracy, outputs["accuracy"])
    _write_csv(per_class, outputs["per_class"])
    _write_csv(speed_rows, outputs["speed"])
    _write_csv(tradeoff, outputs["tradeoff"])
    _write_csv(ensemble_rows, outputs["ensemble"])
    analyze_pareto(metrics_root, metrics_root / "speed", out_root)
    outputs["pareto"] = out_root / "pareto_frontier.csv"
    if (out_root / "object_size_distribution.csv").exists():
        outputs["object_size"] = out_root / "object_size_distribution.csv"

    captions = {
        "accuracy": "Main detection accuracy results.",
        "per_class": "Per-class average precision.",
        "speed": "Speed and efficiency benchmark results.",
        "tradeoff": "Accuracy-efficiency tradeoff.",
        "ensemble": "Ensemble and hybrid prediction outputs.",
        "pareto": "Pareto frontier for accuracy and speed.",
        "object_size": "Object-size distribution.",
    }
    for key, csv_path in outputs.items():
        csv_to_latex(csv_path, paper_root / f"{csv_path.stem}.tex", captions[key], f"tab:{csv_path.stem}")
        target = paper_root / csv_path.name
        target.write_text(Path(csv_path).read_text(encoding="utf-8"), encoding="utf-8")

    # Per-dataset splits: headline accuracy and per-class AP (class sets differ per dataset).
    present = [d for d in KNOWN_DATASETS if any(infer_dataset(r) == d for r in accuracy)]
    for dataset in present:
        acc_rows = [r for r in tradeoff if infer_dataset(r) == dataset]
        pc_rows = [r for r in per_class if infer_dataset(r) == dataset]
        acc_path = out_root / f"headline_{dataset}.csv"
        pc_path = out_root / f"per_class_{dataset}.csv"
        _write_csv(acc_rows, acc_path)
        _write_csv(pc_rows, pc_path)
        outputs[f"headline_{dataset}"] = acc_path
        outputs[f"per_class_{dataset}"] = pc_path
        for path, caption in ((acc_path, f"Headline results on {dataset.upper()}."),
                              (pc_path, f"Per-class AP on {dataset.upper()}.")):
            csv_to_latex(path, paper_root / f"{path.stem}.tex", caption, f"tab:{path.stem}")
            (paper_root / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    manifest = {key: str(path) for key, path in outputs.items()}
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
