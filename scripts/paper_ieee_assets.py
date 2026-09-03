#!/usr/bin/env python3
"""Paper-ready Table I body and Fig. 1 for the IEEE (TELFOR) version.

Both artefacts are generated from the CSVs written by evaluate_slicing.py and
benchmark_slicing.py and are never hand-edited.

    python scripts/paper_ieee_assets.py            # -> paper_ieee/tables/table1_body.tex,
                                                  #    paper_ieee/figures/fig1_dota_latency_map.pdf
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
from matplotlib.lines import Line2D  # noqa: E402

MODEL_NAME = {"yolov8m": "YOLOv8m", "yolo11m": "YOLO11m", "yolo26m": "YOLO26m"}
# Okabe-Ito, colour-blind safe.
MODEL_COLOR = {"yolov8m": "#0072B2", "yolo11m": "#E69F00", "yolo26m": "#009E73"}
# Table I row selection (per model) and display order.
DOTA_ROWS = ["native1024", "native1536", "tile512_ov10", "tile640_ov10", "tile1024_ov10", "tile1024_ov20"]
SHIP_ROWS = ["native640", "native1024", "tile512_ov10", "tile640_ov20"]
SHIP_MODEL = "yolo11m"


def _read(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _f(row: dict, key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def _family(model: str) -> str:
    return model.split("_")[0]


def _variant_label(v: str) -> str:
    if v.startswith("native"):
        return f"native {v[len('native'):]}"
    tile, ov = v.replace("tile", "").split("_ov")
    return f"tile {tile}, {int(ov)}\\%"


def load(ds: str, tables_dir: Path, speed_dir: Path) -> dict[tuple[str, str], dict]:
    metrics = _read(tables_dir / f"{ds}_slicing_metrics.csv")
    speed = {(r["model"], r["variant"]): r for r in _read(speed_dir / ds / f"{ds}_speed.csv")}
    out = {}
    for m in metrics:
        key = (_family(m["model"]), m["variant"])
        out[key] = {**speed.get((m["model"], m["variant"]), {}), **m}
    return out


def table_body(dota: dict, ships: dict, out_tex: Path) -> None:
    def fmt(row, key, spec="{:.3f}"):
        v = _f(row, key)
        return "--" if v is None else spec.format(v)

    def line(row, variant):
        return " & ".join([
            _variant_label(variant), fmt(row, "map50"), fmt(row, "map50_95"), fmt(row, "ap_small"),
            fmt(row, "ap_large"), fmt(row, "precision"), fmt(row, "recall"),
            fmt(row, "latency_p50_ms", "{:.0f}"), fmt(row, "duplicate_box_rate", "{:.2f}"),
        ]) + r" \\"

    lines = [
        r"\begin{tabular}{@{}lrrrrrrrr@{}}",
        r"\toprule",
        r"Variant & mAP@50 & mAP & AP$_S$ & AP$_L$ & P & R & ms & Dup \\",
    ]
    for fam in ("yolov8m", "yolo11m", "yolo26m"):
        lines += [r"\midrule", rf"\multicolumn{{9}}{{@{{}}l}}{{\textit{{DOTA-v1.5, {MODEL_NAME[fam]} (trained at 1024\,px)}}}} \\"]
        for v in DOTA_ROWS:
            row = dota.get((fam, v))
            if row is None:
                continue
            lines.append(line(row, v))
    lines += [r"\midrule", rf"\multicolumn{{9}}{{@{{}}l}}{{\textit{{Ships, {MODEL_NAME[SHIP_MODEL]} (trained at 640\,px)}}}} \\"]
    for v in SHIP_ROWS:
        row = ships.get((SHIP_MODEL, v))
        if row is None:
            continue
        lines.append(line(row, v))
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines), encoding="utf-8")


def fig1(dota: dict, out_pdf: Path) -> None:
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        # Liberation Serif is metrically Times-compatible and a real TrueType file, so
        # fonttype 42 embeds a clean subset. "Nimbus Roman" resolves to an OpenType/CFF
        # face here, which embeds as CID Type 0C and makes pdffonts report a mismatch
        # between font type and embedded file, a needless risk at IEEE PDF eXpress.
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.5, "axes.linewidth": 0.6,
    })
    marker_of = {"native": "o", "tile512": "^", "tile640": "s", "tile1024": "D"}
    fig, ax = plt.subplots(figsize=(3.4, 2.45))
    for (fam, variant), row in sorted(dota.items()):
        x, y = _f(row, "latency_p50_ms"), _f(row, "map50_95")
        if x is None or y is None:
            continue
        kind = "native" if variant.startswith("native") else variant.split("_")[0]
        hollow = variant.endswith("ov20")
        ax.scatter([x], [y], s=22, marker=marker_of[kind],
                   facecolors="none" if hollow else MODEL_COLOR[fam],
                   edgecolors=MODEL_COLOR[fam], linewidths=0.9, zorder=3)
        if kind == "native" and fam == "yolo11m":
            # Centred below the marker: the three detectors sit within a few ms of each
            # other at each native size, so a label offset to the right lands on top of a
            # neighbouring point.
            ax.annotate(variant[len("native"):], (x, y), xytext=(0, -10),
                        textcoords="offset points", fontsize=6.5, color="#333333",
                        ha="center", va="top")
    ax.set_xlabel("Median latency per image (ms, batch 1)")
    ax.set_ylabel("mAP@[50:95]")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    model_handles = [Line2D([], [], color=MODEL_COLOR[f], marker="o", linestyle="", markersize=4, label=MODEL_NAME[f])
                     for f in ("yolov8m", "yolo11m", "yolo26m")]
    policy_handles = [Line2D([], [], color="#444444", marker=m, linestyle="", markersize=4,
                             markerfacecolor="#444444", label=lab)
                      for m, lab in (("o", "native"), ("^", "tile 512"), ("s", "tile 640"), ("D", "tile 1024"))]
    policy_handles.append(Line2D([], [], color="#444444", marker="D", linestyle="", markersize=4,
                                 markerfacecolor="none", label="overlap 20%"))
    leg1 = ax.legend(handles=model_handles, loc="lower right", frameon=False, handletextpad=0.2)
    ax.add_artist(leg1)
    ax.legend(handles=policy_handles, loc="upper left", frameon=False, handletextpad=0.2, ncol=1)
    fig.tight_layout(pad=0.3)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-dir", default=str(ROOT / "results" / "tables"))
    parser.add_argument("--speed-dir", default=str(ROOT / "results" / "metrics" / "speed"))
    parser.add_argument("--out-dir", default=str(ROOT / "paper_ieee"))
    args = parser.parse_args()
    tables_dir, speed_dir, out = Path(args.tables_dir), Path(args.speed_dir), Path(args.out_dir)
    dota = load("dota", tables_dir, speed_dir)
    ships = load("ships", tables_dir, speed_dir)
    table_body(dota, ships, out / "tables" / "table1_body.tex")
    fig1(dota, out / "figures" / "fig1_dota_latency_map.pdf")
    print("wrote", out / "tables" / "table1_body.tex", "and", out / "figures" / "fig1_dota_latency_map.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
