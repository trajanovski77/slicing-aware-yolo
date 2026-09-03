#!/usr/bin/env python3
"""Figure 1 (accuracy-latency Pareto) and Figure 2 (small-object AP by variant).

    python scripts/generate_slicing_figures.py --datasets "dota ships"
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.reporting.pareto import is_dominated  # noqa: E402
from src.utils.logging import configure_logging  # noqa: E402
from src.utils.paths import ensure_dir  # noqa: E402


def _read(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _merge(metrics: list[dict], speed: list[dict]) -> list[dict]:
    speed_by = {(r["model"], r["variant"]): r for r in speed}
    rows = []
    for m in metrics:
        s = speed_by.get((m["model"], m["variant"]), {})
        row = dict(m)
        for k in ("latency_mean_ms", "latency_p50_ms", "fps"):
            row[k] = s.get(k)
        rows.append(row)
    return rows


def _f(row, key):
    try:
        return float(row.get(key))
    except (TypeError, ValueError):
        return None


def _pareto_flags(rows: list[dict]) -> list[bool]:
    """Front = higher mAP@[50:95] at lower (better) median latency."""
    flags = []
    for cand in rows:
        cm, cl = _f(cand, "map50_95"), _f(cand, "latency_p50_ms")
        dominated = any(
            _f(o, "map50_95") is not None and _f(o, "latency_p50_ms") is not None
            and _f(o, "map50_95") >= cm and _f(o, "latency_p50_ms") <= cl
            and (_f(o, "map50_95") > cm or _f(o, "latency_p50_ms") < cl)
            for o in rows if o is not cand)
        flags.append(not dominated)
    return flags


def pareto_figure(rows: list[dict], out_path: Path, title: str) -> None:
    usable = [r for r in rows if _f(r, "latency_p50_ms") and _f(r, "map50_95") is not None and _f(r, "fps")]
    if not usable:
        return
    flags = _pareto_flags(usable)
    plt.figure(figsize=(6, 4.2))
    for model in sorted({r["model"] for r in usable}):
        pts = [(r, flag) for r, flag in zip(usable, flags) if r["model"] == model]
        xs = [_f(r, "latency_p50_ms") for r, _ in pts]
        ys = [_f(r, "map50_95") for r, _ in pts]
        plt.scatter(xs, ys, s=42, label=model)
        for (r, on_front) in pts:
            marker = "*" if r["kind"] == "sliced" else "o"
            plt.scatter([_f(r, "latency_p50_ms")], [_f(r, "map50_95")],
                        s=140 if on_front else 0, facecolors="none",
                        edgecolors="k" if on_front else "none", marker=marker, linewidths=1.2)
            plt.annotate(r["variant"], (_f(r, "latency_p50_ms"), _f(r, "map50_95")),
                         fontsize=6, xytext=(3, 3), textcoords="offset points")
    plt.xlabel("Median latency per image (ms, batch 1)")
    plt.ylabel("mAP@[50:95]")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(out_path.with_suffix(f".{ext}"), dpi=200)
    plt.close()


def small_object_figure(rows: list[dict], out_path: Path, title: str) -> None:
    usable = [r for r in rows if _f(r, "ap_small") is not None]
    if not usable:
        return
    models = sorted({r["model"] for r in usable})
    variants = sorted({r["variant"] for r in usable}, key=lambda v: (not v.startswith("native"), v))
    plt.figure(figsize=(7, 4.2))
    width = 0.8 / max(1, len(models))
    for i, model in enumerate(models):
        by_var = {r["variant"]: _f(r, "ap_small") for r in usable if r["model"] == model}
        ys = [by_var.get(v, 0.0) or 0.0 for v in variants]
        xs = [j + i * width for j in range(len(variants))]
        plt.bar(xs, ys, width=width, label=model)
    plt.xticks([j + 0.4 for j in range(len(variants))], variants, rotation=30, ha="right", fontsize=7)
    plt.ylabel("AP$_{small}$ (COCO)")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(out_path.with_suffix(f".{ext}"), dpi=200)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate slicing Pareto + small-object figures.")
    parser.add_argument("--datasets", default="dota ships")
    parser.add_argument("--tables-dir", default=str(ROOT / "results" / "tables"))
    parser.add_argument("--speed-dir", default=str(ROOT / "results" / "metrics" / "speed"))
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "figures"))
    parser.add_argument("--paper-dir", default=str(ROOT / "paper_assets" / "figures"))
    args = parser.parse_args()
    logger = configure_logging()

    out_dir = ensure_dir(Path(args.output_dir))
    paper_dir = ensure_dir(Path(args.paper_dir))
    for ds in args.datasets.split():
        metrics = _read(Path(args.tables_dir) / f"{ds}_slicing_metrics.csv")
        speed = _read(Path(args.speed_dir) / ds / f"{ds}_speed.csv")
        if not metrics:
            logger.warning("No metrics for %s; skipping figures", ds)
            continue
        rows = _merge(metrics, speed)
        pareto_figure(rows, out_dir / f"slicing-pareto-{ds}", f"Accuracy-latency Pareto ({ds.upper()})")
        pareto_figure(rows, paper_dir / f"slicing-pareto-{ds}", f"Accuracy-latency Pareto ({ds.upper()})")
        small_object_figure(rows, out_dir / f"slicing-small-object-{ds}", f"Small-object AP ({ds.upper()})")
        small_object_figure(rows, paper_dir / f"slicing-small-object-{ds}", f"Small-object AP ({ds.upper()})")
        logger.info("Figures for %s -> %s", ds, paper_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
