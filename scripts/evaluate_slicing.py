#!/usr/bin/env python3
"""Score every prediction file (native + sliced) against the COCO GT.

One evaluator for all variants -> apples-to-apples. Writes a combined metrics CSV and a
per-class AP CSV under results/tables/.

    python scripts/evaluate_slicing.py --config configs/slicing/dota.yaml
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.coco_eval import DEFAULT_MAX_DETS, pr_f1, run_coco_eval  # noqa: E402
from src.eval.diagnostics import ImageMergeCounts, duplicate_rate  # noqa: E402
from src.eval.predictions import load_coco_gt  # noqa: E402
from src.inference.runner import load_slicing_config  # noqa: E402
from src.utils.logging import configure_logging  # noqa: E402
from src.utils.paths import ensure_dir  # noqa: E402

_METRIC_COLS = [
    "dataset", "model", "variant", "kind", "imgsz",
    "map50", "map50_95", "ap_small", "ap_medium", "ap_large",
    "ar_max", "ar_small", "ar_medium", "ar_large",
    "precision", "recall", "f1", "max_dets",
    "tiles_per_image", "mean_dets_per_image_merged", "duplicate_box_rate", "n_detections",
]


def _write_csv(rows: list[dict], path: Path, columns: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="COCO-evaluate native + sliced predictions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gt", default=None)
    parser.add_argument("--pred-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-dets", type=int, default=DEFAULT_MAX_DETS,
                        help="COCO per-image-category detection cap (pycocotools default 100 "
                             "truncates dense DOTA scenes; 100 reproduces stock numbers).")
    args = parser.parse_args()
    logger = configure_logging()

    config = load_slicing_config(args.config)
    name = Path(args.config).stem
    split = str(config.get("split", "val"))
    gt_path = args.gt or (ROOT / "results" / "coco" / f"{name}_{split}_gt.json")
    gt = load_coco_gt(gt_path)
    pred_dir = Path(args.pred_dir or (ROOT / "results" / "predictions" / name))
    out_dir = ensure_dir(Path(args.output_dir or (ROOT / "results" / "tables")))

    metric_rows: list[dict] = []
    per_class_rows: list[dict] = []
    for pred_path in sorted(pred_dir.glob("*.json")):
        payload = load_coco_gt(pred_path)  # generic json loader
        dt = payload.get("dt", [])
        counts = [ImageMergeCounts(c["stem"], c["raw"], c["merged"], c["n_tiles"]) for c in payload.get("counts", [])]
        dup = duplicate_rate(counts)
        coco = run_coco_eval(gt, dt, max_dets=args.max_dets)
        # COCO returns -1 for an area range with no GT (e.g. AP_small on a dataset with no
        # small objects). Blank it so tables/figures show "--" instead of a bogus -1.
        coco = {k: ("" if v == -1.0 else v) for k, v in coco.items() if k != "per_class"} | {"per_class": coco["per_class"]}
        prf = pr_f1(gt, dt, iou_threshold=0.5)
        row = {
            "dataset": name, "model": payload["model"], "variant": payload["variant"],
            "kind": payload["kind"], "imgsz": payload.get("imgsz"),
            "precision": prf["precision"], "recall": prf["recall"], "f1": prf["f1"],
            "tiles_per_image": dup["mean_tiles_per_image"],
            "mean_dets_per_image_merged": dup["mean_dets_per_image_merged"],
            "duplicate_box_rate": dup["duplicate_box_rate"],
            "n_detections": payload.get("n_detections", len(dt)),
            **{k: coco[k] for k in ("map50", "map50_95", "ap_small", "ap_medium", "ap_large",
                                    "ar_max", "ar_small", "ar_medium", "ar_large", "max_dets")},
        }
        metric_rows.append(row)
        for pc in coco["per_class"]:
            per_class_rows.append({"model": payload["model"], "variant": payload["variant"], **pc})
        logger.info("%s / %s: mAP50=%.4f mAP=%.4f AP_s=%.4f R=%.3f dup=%.3f",
                    payload["model"], payload["variant"], coco["map50"], coco["map50_95"],
                    coco["ap_small"], prf["recall"], dup["duplicate_box_rate"])

    metric_rows.sort(key=lambda r: (r["model"], r["kind"], str(r["variant"])))
    _write_csv(metric_rows, out_dir / f"{name}_slicing_metrics.csv", _METRIC_COLS)
    _write_csv(per_class_rows, out_dir / f"{name}_per_class.csv",
               ["model", "variant", "class_id", "class_name", "ap50_95", "ap50"])
    logger.info("Wrote %d metric rows -> %s", len(metric_rows), out_dir / f"{name}_slicing_metrics.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
