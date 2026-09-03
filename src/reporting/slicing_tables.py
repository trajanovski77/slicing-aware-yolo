"""Table 1: native vs sliced per (model, variant), joining accuracy + latency.

Reads the two CSVs written by evaluate_slicing.py and benchmark_slicing.py and produces a
single tidy table (CSV + LaTeX via the reused ``latex.csv_to_latex``).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.reporting.latex import csv_to_latex
from src.utils.paths import ensure_dir

# Column -> header label + float format for the paper table.
_DISPLAY = [
    ("model", "Model", None),
    ("variant", "Variant", None),
    ("map50", "mAP@50", "{:.3f}"),
    ("map50_95", "mAP@[50:95]", "{:.3f}"),
    ("ap_small", "AP$_S$", "{:.3f}"),
    ("ap_medium", "AP$_M$", "{:.3f}"),
    ("ap_large", "AP$_L$", "{:.3f}"),
    ("recall", "R", "{:.3f}"),
    ("f1", "F1", "{:.3f}"),
    ("tiles_per_image", "Tiles", "{:.1f}"),
    ("latency_p50_ms", "Latency (ms)", "{:.1f}"),  # median: robust to DOTA's size-skewed mean
    ("fps_p50", "FPS", "{:.1f}"),
    ("duplicate_box_rate", "Dup", "{:.3f}"),
]


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("model", "")), str(row.get("variant", "")))


def _fmt(value: Any, spec: str | None) -> str:
    if value in (None, ""):
        return "--"
    if spec is None:
        return str(value)
    try:
        return spec.format(float(value))
    except (TypeError, ValueError):
        return str(value)


def build_table1(metrics_csv: str | Path, speed_csv: str | Path) -> list[dict[str, str]]:
    metrics = _read_csv(metrics_csv)
    speed = {_key(r): r for r in _read_csv(speed_csv)}
    rows: list[dict[str, str]] = []
    for m in metrics:
        merged = {**speed.get(_key(m), {}), **m}
        p50 = merged.get("latency_p50_ms")
        if p50 not in (None, ""):
            try:
                merged["fps_p50"] = 1000.0 / float(p50)
            except (TypeError, ValueError):
                pass
        rows.append({label: _fmt(merged.get(col), spec) for col, label, spec in _DISPLAY})
    # native rows first, then sliced, stable within model.
    order = {r["Variant"]: i for i, r in enumerate(rows)}
    rows.sort(key=lambda r: (r["Model"], not r["Variant"].startswith("native"), order[r["Variant"]]))
    return rows


def write_table1(
    metrics_csv: str | Path, speed_csv: str | Path, out_csv: str | Path, out_tex: str | Path,
    caption: str, label: str,
) -> None:
    rows = build_table1(metrics_csv, speed_csv)
    ensure_dir(Path(out_csv).parent)
    with Path(out_csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[lab for _, lab, _ in _DISPLAY])
        writer.writeheader()
        writer.writerows(rows)
    csv_to_latex(out_csv, out_tex, caption=caption, label=label)
