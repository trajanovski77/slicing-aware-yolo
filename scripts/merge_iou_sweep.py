#!/usr/bin/env python3
"""Quick merge-IoU sweep (directional): how the cross-tile NMS merge threshold trades
residual duplicates against detection count. Subset, single model, one tile size.

    python scripts/merge_iou_sweep.py --config configs/slicing/dota.yaml \
        --model yolo11m_dota --tile 640 --overlap 0.20 --limit 60 --device 0
"""
from __future__ import annotations
import argparse, sys, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2  # noqa: E402
from src.inference.merge import box_iou  # noqa: E402
from src.inference.runner import load_slicing_config, resolve_images  # noqa: E402
from src.inference.sliced import SlicingPolicy, UltralyticsTilePredictor, sliced_detections  # noqa: E402
from src.utils.logging import configure_logging  # noqa: E402
from src.utils.ultralytics_env import configure_ultralytics_settings  # noqa: E402


def dup_rate(dets, iou_thr=0.5):
    """Fraction of boxes that share an IoU>thr overlap with another same-class box."""
    n = len(dets)
    if n < 2:
        return 0.0
    flagged = [False] * n
    for i in range(n):
        for j in range(i + 1, n):
            if dets[i].class_id == dets[j].class_id and box_iou(dets[i].xyxy, dets[j].xyxy) > iou_thr:
                flagged[i] = flagged[j] = True
    return sum(flagged) / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="yolo11m_dota")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.20)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--device", default="0")
    ap.add_argument("--merge-ious", default="0.45,0.55,0.65,0.75")
    args = ap.parse_args()
    log = configure_logging()
    configure_ultralytics_settings()
    from ultralytics import YOLO

    cfg = load_slicing_config(args.config)
    conf, iou = float(cfg.get("conf", 0.25)), float(cfg.get("iou", 0.7))
    weights = next(m["weights"] for m in cfg["models"] if m["name"] == args.model)
    images = resolve_images(cfg)[: args.limit]
    model = YOLO(weights, task="detect")
    predictor = UltralyticsTilePredictor(model, imgsz=args.tile, conf=conf, iou=iou, device=args.device)

    print(f"model={args.model} tile={args.tile} overlap={args.overlap} images={len(images)}")
    print(f"{'merge_iou':>9} {'mean_dets':>9} {'dup_rate':>9}")
    for m_iou in (float(x) for x in args.merge_ious.split(",")):
        policy = SlicingPolicy(tile_size=args.tile, overlap=args.overlap,
                               merge_method="nms_class_aware", merge_iou=m_iou)
        counts, dups = [], []
        for p in images:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            dets = sliced_detections(img, policy, predictor).detections
            counts.append(len(dets)); dups.append(dup_rate(dets))
        print(f"{m_iou:>9.2f} {statistics.mean(counts):>9.1f} {statistics.mean(dups):>9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
