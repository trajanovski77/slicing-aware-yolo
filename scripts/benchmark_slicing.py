#!/usr/bin/env python3
"""End-to-end latency for native + sliced variants (batch 1). Run on a QUIET GPU.

    # one card, one dataset (schedule a quiet window):
    python scripts/benchmark_slicing.py --config configs/slicing/dota.yaml --model yolo11m_dota --device 0

    # shared card: cycle every (model, variant) pair per image so a drifting competing
    # load cannot be attributed to whichever block ran first
    python scripts/benchmark_slicing.py --config configs/slicing/dota.yaml --device 0 --interleave

Either way the concurrent utilisation of the target card is sampled during the sweep and
stored beside the timings, so a sweep taken on a busy card can be recognised as
provisional rather than mistaken for a quiet-card measurement.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.interleaved_speed import (  # noqa: E402
    GpuLoadSampler, build_runners, interleaved_sweep, results_from,
)
from src.experiments.sliced_speed import _load_images, benchmark_native, benchmark_sliced  # noqa: E402
from src.inference.runner import iter_variants, load_slicing_config, resolve_images  # noqa: E402
from src.utils.hardware import collect_environment  # noqa: E402
from src.utils.logging import configure_logging  # noqa: E402
from src.utils.paths import ensure_dir, write_json  # noqa: E402
from src.utils.ultralytics_env import configure_ultralytics_settings  # noqa: E402

_COLS = ["dataset", "model", "variant", "mode", "tiles_per_image", "latency_mean_ms",
         "latency_std_ms", "latency_p50_ms", "latency_p95_ms", "fps", "image_count", "iterations",
         "sweep_order", "concurrent_util_median_pct"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Latency benchmark for native + sliced.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--image-count", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--interleave", action="store_true",
                        help="cycle every (model, variant) pair once per image instead of "
                             "timing each pair to completion; use on a shared card")
    args = parser.parse_args()

    logger = configure_logging()
    configure_ultralytics_settings()
    from ultralytics import YOLO

    config = load_slicing_config(args.config)
    name = Path(args.config).stem
    conf = float(config.get("conf", 0.25))
    iou = float(config.get("iou", 0.7))
    max_det = int(config.get("max_det", 10000))   # same cap as the accuracy runs
    images = resolve_images(config)
    variants = list(iter_variants(config))
    out_dir = ensure_dir(Path(args.output_dir or (ROOT / "results" / "metrics" / "speed" / name)))

    models = config.get("models", [])
    if args.model:
        models = [m for m in models if m["name"] == args.model]

    available = [m for m in models if Path(m["weights"]).exists()]
    for model_cfg in models:
        if model_cfg not in available:
            logger.warning("Skipping %s; missing weights", model_cfg["name"])

    rows: list[dict] = []
    with GpuLoadSampler(args.device) as load:
        if args.interleave:
            # Images are read once and shared by every pair; the sequential path re-reads
            # them per pair.
            frames = _load_images(images, args.image_count)
            loaded = [(cfg, YOLO(cfg["weights"], task="detect")) for cfg in available]
            runners = build_runners(loaded, variants, frames, conf, iou, args.device,
                                    args.half, 8, max_det)
            logger.info("Interleaved sweep: %d pairs x %d timed rounds on %d images",
                        len(runners), args.iterations, len(frames))
            interleaved_sweep(
                runners, frames, args.warmup, args.iterations,
                on_round=lambda done, total: logger.info("round %d/%d", done, total)
                if done % 10 == 0 or done == total else None,
            )
            for result in results_from(runners, len(frames), args.warmup):
                result.update({"dataset": name, "sweep_order": "interleaved"})
                write_json(result, out_dir / f"{result['model']}__{result['variant']}.json")
                rows.append(result)
                logger.info("%s / %s: median %.2f ms, mean %.2f ms (%.1f tiles/img)",
                            result["model"], result["variant"], result["latency_p50_ms"],
                            result["latency_mean_ms"], result["tiles_per_image"])
        else:
            for model_cfg in available:
                model = YOLO(model_cfg["weights"], task="detect")
                for variant in variants:
                    if variant.kind == "native":
                        result = benchmark_native(
                            model, images, variant.imgsz, conf, iou, args.device, args.half,
                            args.image_count, args.warmup, args.iterations, max_det=max_det)
                    else:
                        result = benchmark_sliced(
                            model, images, variant.policy, variant.policy.tile_size, conf, iou,
                            args.device, args.half, 8, args.image_count, args.warmup, args.iterations,
                            max_det=max_det)
                    result.update({"dataset": name, "model": model_cfg["name"],
                                   "variant": variant.label, "sweep_order": "sequential"})
                    write_json(result, out_dir / f"{model_cfg['name']}__{variant.label}.json")
                    rows.append(result)
                    logger.info("%s / %s: %.2f ms (%.1f FPS), %.1f tiles/img",
                                model_cfg["name"], variant.label, result["latency_mean_ms"],
                                result["fps"], result["tiles_per_image"])

    concurrent = load.summary()
    logger.info("Concurrent GPU load during the sweep: %s", concurrent)
    for result in rows:
        result["concurrent_util_median_pct"] = concurrent.get("util_median_pct")
        result["concurrent_gpu_load"] = concurrent
        write_json(result, out_dir / f"{result['model']}__{result['variant']}.json")

    environment = collect_environment(".")
    environment["concurrent_gpu_load"] = concurrent
    write_json(environment, out_dir / "environment.json")
    csv_path = out_dir / f"{name}_speed.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d speed rows -> %s", len(rows), csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
