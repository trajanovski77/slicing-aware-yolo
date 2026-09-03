#!/usr/bin/env python3
"""Correctness anchor: compare our custom slicer against the SAHI library on YOLOv8m.

Runs both slicers on the same images with matching tile/overlap/merge settings and reports
box-set agreement (matched-at-IoU rate) + detection-count deltas. A high agreement lets the
paper answer "why not just use SAHI?" — our reimplementation matches the reference while
supporting YOLO26 (SAHI's Ultralytics adapter may not).

Requires the optional dep:  pip install sahi

    python scripts/sahi_crosscheck.py --config configs/slicing/dota.yaml \
        --model yolov8m_dota --tile 640 --overlap 0.20 --limit 50 --device 0
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from src.inference.merge import Detection, box_iou  # noqa: E402
from src.inference.runner import load_slicing_config, resolve_images  # noqa: E402
from src.inference.sliced import SlicingPolicy, UltralyticsTilePredictor, sliced_detections  # noqa: E402
from src.utils.logging import configure_logging  # noqa: E402
from src.utils.ultralytics_env import configure_ultralytics_settings  # noqa: E402


def _match_rate(a: list[Detection], b: list[Detection], iou_thr: float) -> float:
    if not a and not b:
        return 1.0
    claimed = [False] * len(b)
    matched = 0
    for det in sorted(a, key=lambda d: d.confidence, reverse=True):
        best_iou, best_j = 0.0, -1
        for j, other in enumerate(b):
            if claimed[j] or other.class_id != det.class_id:
                continue
            iou = box_iou(det.xyxy, other.xyxy)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_thr:
            claimed[best_j] = True
            matched += 1
    return matched / max(len(a), len(b), 1)


def _sahi_predictions(image_path, weights, tile, overlap, conf, iou, device) -> list[Detection]:
    from sahi.predict import get_sliced_prediction

    try:
        from sahi import AutoDetectionModel
        model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=weights,
            confidence_threshold=conf, device=(f"cuda:{device}" if device not in (None, "cpu") else "cpu"))
    except Exception:  # older SAHI naming
        from sahi import AutoDetectionModel
        model = AutoDetectionModel.from_pretrained(
            model_type="yolov8", model_path=weights,
            confidence_threshold=conf, device=(f"cuda:{device}" if device not in (None, "cpu") else "cpu"))

    result = get_sliced_prediction(
        str(image_path), model, slice_height=tile, slice_width=tile,
        overlap_height_ratio=overlap, overlap_width_ratio=overlap,
        postprocess_type="NMS", postprocess_match_metric="IOU", postprocess_match_threshold=iou)
    dets = []
    for obj in result.object_prediction_list:
        x1, y1, x2, y2 = obj.bbox.to_xyxy()
        dets.append(Detection(int(obj.category.id), float(obj.score.value), (x1, y1, x2, y2)))
    return dets


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-check custom slicer vs SAHI.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tile", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--device", default=None)
    parser.add_argument("--match-iou", type=float, default=0.5)
    args = parser.parse_args()
    logger = configure_logging()
    configure_ultralytics_settings()
    from ultralytics import YOLO

    config = load_slicing_config(args.config)
    conf = float(config.get("conf", 0.25))
    weights = next(m["weights"] for m in config["models"] if m["name"] == args.model)
    images = resolve_images(config)[: args.limit]

    model = YOLO(weights, task="detect")
    policy = SlicingPolicy(tile_size=args.tile, overlap=args.overlap,
                           merge_method="nms_class_aware", merge_iou=args.iou)
    predictor = UltralyticsTilePredictor(model, imgsz=args.tile, conf=conf, iou=config.get("iou", 0.7),
                                         device=args.device)

    rates, ours_counts, sahi_counts = [], [], []
    for path in images:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        ours = sliced_detections(image, policy, predictor).detections
        sahi = _sahi_predictions(path, weights, args.tile, args.overlap, conf, args.iou, args.device)
        rates.append(_match_rate(ours, sahi, args.match_iou))
        ours_counts.append(len(ours))
        sahi_counts.append(len(sahi))

    logger.info("Images: %d | mean match-rate @IoU%.2f: %.3f | dets ours=%.1f sahi=%.1f (mean)",
                len(rates), args.match_iou, statistics.mean(rates) if rates else 0.0,
                statistics.mean(ours_counts) if ours_counts else 0.0,
                statistics.mean(sahi_counts) if sahi_counts else 0.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
