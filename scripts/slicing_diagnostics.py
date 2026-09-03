#!/usr/bin/env python3
"""RQ5 diagnostics for sliced predictions: duplicate-box rate + boundary false positives.

Reads the sliced prediction files + COCO GT and writes results/tables/<name>_diagnostics.csv.

    python scripts/slicing_diagnostics.py --config configs/slicing/dota.yaml
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.diagnostics import (  # noqa: E402
    ImageMergeCounts, aggregate_boundary, boundary_false_positives, duplicate_rate,
)
from src.eval.predictions import load_coco_gt  # noqa: E402
from src.inference.merge import Detection  # noqa: E402
from src.inference.runner import iter_variants, load_slicing_config  # noqa: E402
from src.utils.logging import configure_logging  # noqa: E402
from src.utils.paths import ensure_dir  # noqa: E402


def _index_gt(gt):
    dims = {int(im["id"]): (int(im["width"]), int(im["height"])) for im in gt["images"]}
    boxes = defaultdict(list)
    for ann in gt["annotations"]:
        x, y, w, h = ann["bbox"]
        boxes[int(ann["image_id"])].append((int(ann["category_id"]), (x, y, x + w, y + h)))
    return dims, boxes


def main() -> int:
    parser = argparse.ArgumentParser(description="Slicing RQ5 diagnostics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gt", default=None)
    parser.add_argument("--pred-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    logger = configure_logging()

    config = load_slicing_config(args.config)
    name = Path(args.config).stem
    split = str(config.get("split", "val"))
    gt = load_coco_gt(args.gt or (ROOT / "results" / "coco" / f"{name}_{split}_gt.json"))
    dims, gt_boxes = _index_gt(gt)
    pred_dir = Path(args.pred_dir or (ROOT / "results" / "predictions" / name))
    out_dir = ensure_dir(Path(args.output_dir or (ROOT / "results" / "tables")))

    policy_by_label = {v.label: v.policy for v in iter_variants(config) if v.kind == "sliced"}

    rows: list[dict] = []
    for pred_path in sorted(pred_dir.glob("*.json")):
        payload = load_coco_gt(pred_path)
        if payload.get("kind") != "sliced":
            continue
        policy = policy_by_label.get(payload["variant"])
        if policy is None:
            continue
        counts = [ImageMergeCounts(c["stem"], c["raw"], c["merged"], c["n_tiles"]) for c in payload["counts"]]
        dup = duplicate_rate(counts)

        dets_by_image = defaultdict(list)
        for rec in payload["dt"]:
            x, y, w, h = rec["bbox"]
            dets_by_image[int(rec["image_id"])].append(
                Detection(int(rec["category_id"]), float(rec["score"]), (x, y, x + w, y + h))
            )
        per_image = []
        for image_id, dets in dets_by_image.items():
            if image_id not in dims:
                continue
            width, height = dims[image_id]
            per_image.append(boundary_false_positives(
                dets, gt_boxes.get(image_id, []), width, height, policy))
        boundary = aggregate_boundary(per_image)

        rows.append({
            "dataset": name, "model": payload["model"], "variant": payload["variant"],
            "duplicate_box_rate": dup["duplicate_box_rate"],
            "mean_tiles_per_image": dup["mean_tiles_per_image"],
            "mean_dets_per_image_raw": dup["mean_dets_per_image_raw"],
            "mean_dets_per_image_merged": dup["mean_dets_per_image_merged"],
            **boundary,
        })
        logger.info("%s / %s: dup=%.3f boundary_fp_rate=%.3f (of FPs)",
                    payload["model"], payload["variant"], dup["duplicate_box_rate"],
                    boundary["boundary_fp_rate_of_fps"])

    cols = ["dataset", "model", "variant", "duplicate_box_rate", "mean_tiles_per_image",
            "mean_dets_per_image_raw", "mean_dets_per_image_merged",
            "boundary_fp_rate_of_fps", "boundary_fp_rate_of_dets", "near_seam_fraction",
            "near_seam_fp", "total_fp"]
    path = out_dir / f"{name}_diagnostics.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d diagnostic rows -> %s", len(rows), path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
