from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.utils.paths import ensure_dir

# Datasets in the YOLO-family comparison. Order is the canonical reporting order.
KNOWN_DATASETS = ("dota", "dior", "ships", "whu")


def infer_dataset(row: dict[str, Any]) -> str:
    """Best-effort dataset tag for a run, parsed from its run_name (e.g. yolo11m_dior_int8)."""
    explicit = str(row.get("dataset") or "").strip().lower()
    if explicit in KNOWN_DATASETS:
        return explicit
    run_name = str(row.get("run_name", "")).lower()
    for dataset in KNOWN_DATASETS:
        if f"_{dataset}_" in run_name or run_name.endswith(f"_{dataset}") or f"_{dataset}" in run_name:
            return dataset
    return ""


def _mark_per_dataset(
    rows: list[dict[str, Any]],
    marker: Callable[..., list[dict[str, Any]]],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run a frontier marker independently within each dataset so points from different
    datasets (whose mAP / latency are not comparable) never dominate one another."""
    for row in rows:
        row.setdefault("dataset", infer_dataset(row))
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("dataset", "")), []).append(row)
    ordered = [d for d in KNOWN_DATASETS if d in groups] + [d for d in groups if d not in KNOWN_DATASETS]
    marked: list[dict[str, Any]] = []
    for dataset in ordered:
        marked.extend(marker(groups[dataset], **kwargs))
    return marked


def _read_csvs(root: Path, pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in sorted(root.rglob(pattern)):
        if path.stat().st_size == 0:
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def base_run_name(name: str) -> str:
    for suffix in ("_gpu", "_cpu"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def is_dominated(
    candidate: dict[str, Any],
    challenger: dict[str, Any],
    accuracy_key: str = "map50_95",
    speed_key: str = "fps",
) -> bool:
    c_acc = _float(candidate, accuracy_key)
    c_speed = _float(candidate, speed_key)
    o_acc = _float(challenger, accuracy_key)
    o_speed = _float(challenger, speed_key)
    if None in (c_acc, c_speed, o_acc, o_speed):
        return False
    at_least_as_good = o_acc >= c_acc and o_speed >= c_speed
    strictly_better = o_acc > c_acc or o_speed > c_speed
    return at_least_as_good and strictly_better


def mark_pareto_frontier(
    rows: list[dict[str, Any]],
    accuracy_key: str = "map50_95",
    speed_key: str = "fps",
) -> list[dict[str, Any]]:
    marked: list[dict[str, Any]] = []
    acc_values = [_float(row, accuracy_key) for row in rows]
    speed_values = [_float(row, speed_key) for row in rows]
    valid_acc = [value for value in acc_values if value is not None]
    valid_speed = [value for value in speed_values if value is not None]
    max_acc = max(valid_acc) if valid_acc else None
    max_speed = max(valid_speed) if valid_speed else None
    for row in rows:
        output = dict(row)
        output["pareto_efficient"] = not any(
            is_dominated(row, other, accuracy_key, speed_key) for other in rows if other is not row
        )
        acc = _float(row, accuracy_key)
        speed = _float(row, speed_key)
        if acc is not None and speed is not None and max_acc and max_speed:
            output["accuracy_efficiency_score"] = (acc / max_acc) * (speed / max_speed)
        else:
            output["accuracy_efficiency_score"] = None
        marked.append(output)
    return marked


def is_latency_dominated(
    candidate: dict[str, Any],
    challenger: dict[str, Any],
    accuracy_key: str = "map50_95",
    latency_key: str = "latency_mean_ms",
) -> bool:
    c_acc = _float(candidate, accuracy_key)
    c_latency = _float(candidate, latency_key)
    o_acc = _float(challenger, accuracy_key)
    o_latency = _float(challenger, latency_key)
    if None in (c_acc, c_latency, o_acc, o_latency):
        return False
    at_least_as_good = o_acc >= c_acc and o_latency <= c_latency
    strictly_better = o_acc > c_acc or o_latency < c_latency
    return at_least_as_good and strictly_better


def mark_latency_pareto_frontier(
    rows: list[dict[str, Any]],
    accuracy_key: str = "map50_95",
    latency_key: str = "latency_mean_ms",
) -> list[dict[str, Any]]:
    marked: list[dict[str, Any]] = []
    acc_values = [_float(row, accuracy_key) for row in rows]
    latency_values = [_float(row, latency_key) for row in rows]
    valid_acc = [value for value in acc_values if value is not None]
    valid_latency = [value for value in latency_values if value is not None and value > 0]
    max_acc = max(valid_acc) if valid_acc else None
    min_latency = min(valid_latency) if valid_latency else None
    for row in rows:
        output = dict(row)
        output["pareto_efficient"] = not any(
            is_latency_dominated(row, other, accuracy_key, latency_key)
            for other in rows
            if other is not row
        )
        acc = _float(row, accuracy_key)
        latency = _float(row, latency_key)
        if acc is not None and latency is not None and latency > 0 and max_acc and min_latency:
            output["accuracy_efficiency_score"] = (acc / max_acc) * (min_latency / latency)
        else:
            output["accuracy_efficiency_score"] = None
        marked.append(output)
    return marked


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "run_name",
        "dataset",
        "family",
        "scale",
        "task",
        "split",
        "imgsz",
        "batch",
        "half",
        "export_precision",
        "precision_type",
        "map50_95",
        "map50",
        "precision",
        "recall",
        "fps",
        "latency_mean_ms",
        "latency_std_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "parameters",
        "model_size_mb",
        "exported_model",
        "pareto_efficient",
        "accuracy_efficiency_score",
    ]
    ordered = [key for key in preferred if key in columns] + [key for key in columns if key not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def _infer_family_scale(row: dict[str, Any]) -> dict[str, str]:
    run_name = str(row.get("run_name", ""))
    family = row.get("family")
    scale = row.get("scale")
    if family and scale:
        return {"family": str(family), "scale": str(scale)}
    for candidate in ("yolov8", "yolo11", "yolo26"):
        if candidate in run_name:
            suffix = run_name.split(candidate, 1)[1]
            detected = next((char for char in suffix if char in "nsmxl"), "")
            return {"family": candidate, "scale": detected}
    return {"family": "", "scale": ""}


def build_pareto_rows(
    metrics_dir: str | Path = "results/metrics",
    speed_dir: str | Path = "results/metrics/speed",
) -> list[dict[str, Any]]:
    metrics = _read_csvs(Path(metrics_dir), "metrics.csv")
    speed = _read_csvs(Path(speed_dir), "*.csv")
    speed_by_name = {base_run_name(str(row.get("run_name", ""))): row for row in speed}
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        run_name = str(metric.get("run_name", ""))
        merged = dict(metric)
        merged.update({k: v for k, v in _infer_family_scale(metric).items() if not merged.get(k)})
        merged["dataset"] = infer_dataset(merged)
        speed_row = speed_by_name.get(run_name) or speed_by_name.get(base_run_name(run_name))
        if speed_row:
            for key in ("fps", "latency_mean_ms", "latency_std_ms", "parameters", "flops", "model_size_mb", "imgsz", "batch", "half"):
                if key in speed_row:
                    merged[key] = speed_row[key]
        rows.append(merged)
    return rows


def _infer_precision(row: dict[str, Any]) -> str:
    precision = str(row.get("export_precision") or "").lower()
    if precision in {"fp32", "fp16", "int8"}:
        return precision
    precision = str(row.get("precision_type") or "").lower()
    if precision in {"fp32", "fp16", "int8"}:
        return precision
    run_name = str(row.get("run_name", "")).lower()
    for candidate in ("fp32", "fp16", "int8"):
        if run_name.endswith(f"_{candidate}") or f"_{candidate}_" in run_name:
            return candidate
    if str(row.get("half", "")).lower() == "true":
        return "fp16"
    return "fp32"


def _is_quantization_candidate(row: dict[str, Any]) -> bool:
    run_name = str(row.get("run_name", "")).lower()
    if any(run_name.endswith(f"_{precision}") for precision in ("fp32", "fp16", "int8")):
        return True
    if str(row.get("exported_model", "")).strip():
        return True
    return _infer_precision(row) in {"fp16", "int8"}


def build_quantization_pareto_rows(
    metrics_dir: str | Path = "results/metrics",
    speed_dir: str | Path = "results/metrics/speed",
) -> list[dict[str, Any]]:
    metrics = [row for row in _read_csvs(Path(metrics_dir), "metrics.csv") if _is_quantization_candidate(row)]
    speed = [row for row in _read_csvs(Path(speed_dir), "*.csv") if _is_quantization_candidate(row)]
    speed_by_name = {base_run_name(str(row.get("run_name", ""))): row for row in speed}
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        run_name = str(metric.get("run_name", ""))
        merged = dict(metric)
        merged.update({k: v for k, v in _infer_family_scale(metric).items() if not merged.get(k)})
        merged["dataset"] = infer_dataset(merged)
        merged["export_precision"] = _infer_precision(merged)
        merged["precision_type"] = merged["export_precision"]
        speed_row = speed_by_name.get(run_name) or speed_by_name.get(base_run_name(run_name))
        if speed_row:
            for key in (
                "fps",
                "latency_mean_ms",
                "latency_std_ms",
                "latency_p50_ms",
                "latency_p95_ms",
                "parameters",
                "flops",
                "model_size_mb",
                "imgsz",
                "batch",
                "half",
                "export_precision",
                "precision_type",
                "exported_model",
            ):
                if key in speed_row and speed_row[key] not in (None, ""):
                    merged[key] = speed_row[key]
            merged["export_precision"] = _infer_precision(merged)
            merged["precision_type"] = merged["export_precision"]
        rows.append(merged)
    return rows


def analyze_pareto(
    metrics_dir: str | Path = "results/metrics",
    speed_dir: str | Path = "results/metrics/speed",
    output_dir: str | Path = "results/tables",
    accuracy_key: str = "map50_95",
    speed_key: str = "fps",
) -> list[dict[str, Any]]:
    rows = build_pareto_rows(metrics_dir, speed_dir)
    marked = _mark_per_dataset(rows, mark_pareto_frontier, accuracy_key=accuracy_key, speed_key=speed_key)
    _write_csv(marked, Path(output_dir) / "pareto_frontier.csv")
    return marked


def analyze_quantization_pareto(
    metrics_dir: str | Path = "results/metrics",
    speed_dir: str | Path = "results/metrics/speed",
    output_dir: str | Path = "results/tables",
    paper_dir: str | Path = "paper_assets/tables",
    accuracy_key: str = "map50_95",
    latency_key: str = "latency_mean_ms",
) -> list[dict[str, Any]]:
    rows = build_quantization_pareto_rows(metrics_dir, speed_dir)
    marked = _mark_per_dataset(
        rows, mark_latency_pareto_frontier, accuracy_key=accuracy_key, latency_key=latency_key
    )
    output_path = Path(output_dir) / "quantization_pareto.csv"
    paper_path = Path(paper_dir) / "quantization_pareto.csv"
    _write_csv(marked, output_path)
    _write_csv(marked, paper_path)
    return marked
