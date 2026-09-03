#!/usr/bin/env python3
"""Stratified analyses for the paper (CPU only, from saved prediction dumps).

  dota-sizes      : DOTA val scene-size statistics (F06).
  ship-sizes      : ship val image-size distribution and the single-window share (F09/F20).
  ship-strata     : ship metrics on single-window (max side <= 512) vs multi-window images.
  dota-density    : native vs sliced fixed-threshold recall on scenes with <=300 vs >300 GT boxes (C01).

    python scripts/analysis_strata.py dota-sizes ship-sizes ship-strata dota-density
Outputs JSON under results/tables/strata_<name>.json and prints a summary.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.coco_eval import pr_f1, run_coco_eval  # noqa: E402

OUT = ROOT / "results" / "tables"


def _gt(ds: str) -> dict:
    return json.load(open(ROOT / "results" / "coco" / f"{ds}_val_gt.json"))


def _dt(ds: str, model: str, variant: str) -> list[dict]:
    return json.load(open(ROOT / "results" / "predictions" / ds / f"{model}__{variant}.json"))["dt"]


def _subset(gt: dict, ids: set[int]) -> dict:
    return {**gt, "images": [i for i in gt["images"] if i["id"] in ids],
            "annotations": [a for a in gt["annotations"] if a["image_id"] in ids]}


def dota_sizes() -> dict:
    gt = _gt("dota")
    longs = sorted(max(i["width"], i["height"]) for i in gt["images"])
    mp = sorted(i["width"] * i["height"] / 1e6 for i in gt["images"])
    smallest = min(gt["images"], key=lambda i: i["width"] * i["height"])
    largest = max(gt["images"], key=lambda i: i["width"] * i["height"])
    per_img = Counter(a["image_id"] for a in gt["annotations"])
    out = {
        "n_images": len(gt["images"]), "n_boxes": len(gt["annotations"]),
        "smallest": [smallest["width"], smallest["height"]], "largest": [largest["width"], largest["height"]],
        "long_side_min": longs[0], "long_side_median": statistics.median(longs), "long_side_max": longs[-1],
        "mp_median": statistics.median(mp), "mp_mean": statistics.mean(mp),
        "n_long_side_gt_1024": sum(1 for v in longs if v > 1024),
        "n_long_side_gt_2600": sum(1 for v in longs if v > 2600),
        "n_long_side_gt_6200": sum(1 for v in longs if v > 6200),
        "n_images_gt_300_boxes": sum(1 for v in per_img.values() if v > 300),
        "share_boxes_in_dense_scenes": sum(v for v in per_img.values() if v > 300) / len(gt["annotations"]),
        "difficult_share": sum(1 for a in gt["annotations"] if a.get("ignore") or a.get("difficult")) / len(gt["annotations"]),
        "small_share_coco": sum(1 for a in gt["annotations"] if a["area"] < 32 ** 2) / len(gt["annotations"]),
    }
    return out


def ship_sizes() -> dict:
    gt = _gt("ships")
    sizes = Counter((i["width"], i["height"]) for i in gt["images"])
    maxside = [max(i["width"], i["height"]) for i in gt["images"]]
    return {
        "n_images": len(gt["images"]), "size_counts": {f"{w}x{h}": c for (w, h), c in sizes.most_common()},
        "n_max_side_le_512": sum(1 for v in maxside if v <= 512),
        "n_max_side_le_640": sum(1 for v in maxside if v <= 640),
        "n_max_side_gt_768": sum(1 for v in maxside if v > 768),
        "max_side_max": max(maxside),
    }


def ship_strata(model: str = "yolo11m_ships") -> dict:
    gt = _gt("ships")
    small = {i["id"] for i in gt["images"] if max(i["width"], i["height"]) <= 512}
    large = {i["id"] for i in gt["images"]} - small
    out = {"model": model, "n_single_window": len(small), "n_multi_window": len(large), "rows": {}}
    for variant in ("native640", "native1024", "tile512_ov10", "tile640_ov20"):
        dt = _dt("ships", model, variant)
        for name, ids in (("single_window_le512", small), ("multi_window_gt512", large)):
            sub = _subset(gt, ids)
            sdt = [d for d in dt if d["image_id"] in ids]
            coco = run_coco_eval(sub, sdt)
            prf = pr_f1(sub, sdt, iou_threshold=0.5)
            out["rows"][f"{variant}/{name}"] = {
                "map50": coco["map50"], "map50_95": coco["map50_95"], "ap_small": coco["ap_small"],
                "precision": prf["precision"], "recall": prf["recall"], "n_dets": len(sdt),
            }
    return out


def dota_density(model: str = "yolo11m_dota") -> dict:
    gt = _gt("dota")
    per_img = Counter(a["image_id"] for a in gt["annotations"])
    dense = {i for i, n in per_img.items() if n > 300}
    sparse = {i["id"] for i in gt["images"]} - dense
    out = {"model": model, "n_dense_scenes": len(dense), "n_sparse_scenes": len(sparse),
           "boxes_dense": sum(per_img[i] for i in dense), "boxes_sparse": sum(per_img[i] for i in sparse), "rows": {}}
    for variant in ("native640", "native1024", "native1536", "tile512_ov10", "tile640_ov10", "tile1024_ov10", "tile1024_ov20"):
        try:
            dt = _dt("dota", model, variant)
        except FileNotFoundError:
            continue
        for name, ids in (("sparse_le300", sparse), ("dense_gt300", dense)):
            sub = _subset(gt, ids)
            sdt = [d for d in dt if d["image_id"] in ids]
            prf = pr_f1(sub, sdt, iou_threshold=0.5)
            coco = run_coco_eval(sub, sdt)
            out["rows"][f"{variant}/{name}"] = {"recall": prf["recall"], "precision": prf["precision"],
                                                 "map50": coco["map50"], "ap_small": coco["ap_small"], "n_dets": len(sdt)}
    return out


def main() -> int:
    tasks = sys.argv[1:] or ["dota-sizes", "ship-sizes", "ship-strata"]
    fns = {"dota-sizes": dota_sizes, "ship-sizes": ship_sizes, "ship-strata": ship_strata, "dota-density": dota_density}
    for t in tasks:
        res = fns[t]()
        path = OUT / f"strata_{t.replace('-', '_')}.json"
        path.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"### {t} -> {path.name}")
        print(json.dumps(res, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
