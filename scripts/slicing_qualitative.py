#!/usr/bin/env python3
"""Figure 3: native vs sliced side-by-side panels (recovered small objects + boundary dups).

    python scripts/slicing_qualitative.py --config configs/slicing/dota.yaml \
        --model yolo11m_dota --native-imgsz 1024 --tile 640 --overlap 0.20 \
        --samples 6 --device 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from src.inference.native import native_detections  # noqa: E402
from src.inference.runner import load_slicing_config, resolve_images  # noqa: E402
from src.inference.sliced import SlicingPolicy, UltralyticsTilePredictor, sliced_detections  # noqa: E402
from src.utils.logging import configure_logging  # noqa: E402
from src.utils.paths import ensure_dir  # noqa: E402
from src.utils.ultralytics_env import configure_ultralytics_settings  # noqa: E402


def _draw(image, detections, color):
    canvas = image.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    return canvas


def _resize_to_width(image, width):
    h, w = image.shape[:2]
    if w <= width:
        return image
    return cv2.resize(image, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


def _label(image, text):
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(image, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description="Native vs sliced qualitative panels.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--native-imgsz", type=int, default=1024)
    parser.add_argument("--tile", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--panel-width", type=int, default=900)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    logger = configure_logging()
    configure_ultralytics_settings()
    from ultralytics import YOLO

    config = load_slicing_config(args.config)
    name = Path(args.config).stem
    conf = float(config.get("conf", 0.25))
    iou = float(config.get("iou", 0.7))
    weights = next(m["weights"] for m in config["models"] if m["name"] == args.model)
    images = resolve_images(config)[: args.samples]
    out_dir = ensure_dir(Path(args.output_dir or (ROOT / "results" / "predictions" / name / "qualitative")))

    model = YOLO(weights, task="detect")
    policy = SlicingPolicy(tile_size=args.tile, overlap=args.overlap,
                           merge_method=config.get("merge", {}).get("method", "nms_class_aware"),
                           merge_iou=config.get("merge", {}).get("iou", 0.55))
    predictor = UltralyticsTilePredictor(model, imgsz=args.tile, conf=conf, iou=iou, device=args.device)

    for path in images:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        native = native_detections(model, image, args.native_imgsz, conf, iou, args.device)
        sliced = sliced_detections(image, policy, predictor).detections
        left = _label(_resize_to_width(_draw(image, native, (0, 0, 255)), args.panel_width),
                      f"native@{args.native_imgsz}  ({len(native)} det)")
        right = _label(_resize_to_width(_draw(image, sliced, (0, 200, 0)), args.panel_width),
                       f"sliced {policy.label}  ({len(sliced)} det)")
        h = min(left.shape[0], right.shape[0])
        panel = cv2.hconcat([left[:h], right[:h]])
        out_path = out_dir / f"{args.model}__{path.stem}.jpg"
        cv2.imwrite(str(out_path), panel)
        logger.info("panel %s: native=%d sliced=%d", path.stem, len(native), len(sliced))
    logger.info("Wrote panels -> %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
