#!/usr/bin/env python3
"""Run native + sliced inference for a slicing config and dump COCO detections.

For every (model x variant) it writes one prediction file under
``results/predictions/<dataset>/<model>__<variant>.json`` containing the COCO ``dt``
records (original-image coords) and per-image merge counts (for RQ5 diagnostics).

    python scripts/predict_slicing.py --config configs/slicing/dota.yaml --device 0
    python scripts/predict_slicing.py --config configs/slicing/dota.yaml --model yolo11m_dota --limit 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from src.eval.predictions import detections_to_coco, image_id_index, load_coco_gt, write_coco_dt  # noqa: E402
from src.inference.native import native_detections  # noqa: E402
from src.inference.runner import iter_variants, load_slicing_config, resolve_images  # noqa: E402
from src.inference.sliced import UltralyticsTilePredictor, sliced_detections  # noqa: E402
from src.utils.logging import configure_logging  # noqa: E402
from src.utils.paths import ensure_dir, write_json  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.utils.ultralytics_env import configure_ultralytics_settings  # noqa: E402


def _run_variant(model, variant, images, conf, iou, device, per_stem, max_det=10000):
    """Return (per_image_detections{stem: [Detection]}, counts[list])."""
    per_image: dict[str, list] = {}
    counts: list[dict] = []
    if variant.kind == "sliced":
        predictor = UltralyticsTilePredictor(
            model, imgsz=variant.policy.tile_size, conf=conf, iou=iou, device=device,
            max_det=max_det,
        )
    for path in images:
        stem = path.stem
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        if variant.kind == "native":
            dets = native_detections(model, image, variant.imgsz, conf, iou, device, max_det=max_det)
            per_image[stem] = dets
            counts.append({"stem": stem, "raw": len(dets), "merged": len(dets), "n_tiles": 1})
        else:
            result = sliced_detections(image, variant.policy, predictor)
            per_image[stem] = result.detections
            counts.append({
                "stem": stem, "raw": result.raw_detection_count,
                "merged": result.merged_detection_count, "n_tiles": result.n_tiles,
            })
    return per_image, counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Native + sliced inference -> COCO detections.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gt", default=None, help="COCO GT json (default results/coco/<name>_<split>_gt.json)")
    parser.add_argument("--model", default=None, help="Run only this model name (default: all).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=0, help="Limit #images (smoke runs).")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--variants", default=None,
                        help="Comma-separated variant labels to run (default: all), e.g. native640,native1024")
    parser.add_argument("--max-det", type=int, default=None,
                        help="Per-image (native) / per-tile (sliced) detection cap; overrides config max_det "
                             "(default 10000; Ultralytics' own default of 300 truncates dense DOTA scenes).")
    args = parser.parse_args()

    logger = configure_logging()
    set_seed(args.seed)
    configure_ultralytics_settings()
    from ultralytics import YOLO

    config = load_slicing_config(args.config)
    name = Path(args.config).stem
    split = str(config.get("split", "val"))
    gt_path = args.gt or (ROOT / "results" / "coco" / f"{name}_{split}_gt.json")
    stem_to_id = image_id_index(load_coco_gt(gt_path))

    conf = float(config.get("conf", 0.25))
    iou = float(config.get("iou", 0.7))
    max_det = int(args.max_det if args.max_det is not None else config.get("max_det", 10000))
    images = resolve_images(config)
    if args.limit:
        images = images[: args.limit]
    logger.info("Dataset %s: %d images, GT=%s", name, len(images), gt_path)

    out_dir = ensure_dir(Path(args.output_dir or (ROOT / "results" / "predictions" / name)))
    variants = list(iter_variants(config))
    if args.variants:
        wanted = {v.strip() for v in args.variants.split(",") if v.strip()}
        variants = [v for v in variants if v.label in wanted]
        missing = wanted - {v.label for v in variants}
        if missing:
            logger.error("Unknown variant label(s): %s", ", ".join(sorted(missing)))
            return 2
    logger.info("Variants: %s | conf=%.2f iou=%.2f max_det=%d", [v.label for v in variants], conf, iou, max_det)
    models = config.get("models", [])
    if args.model:
        models = [m for m in models if m["name"] == args.model]
        if not models:
            logger.error("Model %s not in config", args.model)
            return 2

    for model_cfg in models:
        weights = model_cfg["weights"]
        if not Path(weights).exists():
            logger.warning("Skipping %s; missing weights %s", model_cfg["name"], weights)
            continue
        model = YOLO(weights, task="detect")
        for variant in variants:
            per_image, counts = _run_variant(model, variant, images, conf, iou, args.device, stem_to_id, max_det)
            dt = detections_to_coco(per_image, stem_to_id)
            payload = {
                "dataset": name, "model": model_cfg["name"], "variant": variant.label,
                "kind": variant.kind, "imgsz": variant.imgsz,
                "policy": (variant.policy.label if variant.policy else None),
                "conf": conf, "iou": iou, "max_det": max_det,
                "n_images": len(counts), "n_detections": len(dt),
                "dt": dt, "counts": counts,
            }
            out_path = out_dir / f"{model_cfg['name']}__{variant.label}.json"
            write_json(payload, out_path)
            logger.info("%s / %s: %d detections -> %s",
                        model_cfg["name"], variant.label, len(dt), out_path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
