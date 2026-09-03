"""End-to-end latency for native vs. sliced inference (batch 1, quiet GPU).

Follows the prior repo's ``speed.py`` methodology — warmup, many timed iterations,
mean/std/p50/p95/FPS — but the sliced timer wraps the *whole* per-image pipeline (tiling +
N tile forward passes + merge), and also reports tiles-per-image, the efficiency figure the
proposal asks for. Latency is a trade-off shape on the logged hardware, not a portable
throughput claim.
"""

from __future__ import annotations

import random
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from src.inference.native import native_detections
from src.inference.sliced import SlicingPolicy, UltralyticsTilePredictor, sliced_detections, windows_for_image


def summarize_latency(times_ms: Sequence[float]) -> dict[str, float]:
    times = list(times_ms)
    mean_ms = statistics.mean(times)
    return {
        "latency_mean_ms": mean_ms,
        "latency_std_ms": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "latency_p50_ms": float(np.percentile(times, 50)),
        "latency_p95_ms": float(np.percentile(times, 95)),
        "fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
        "iterations": len(times),
    }


def _load_images(paths: Sequence[str | Path], limit: int) -> list[np.ndarray]:
    ordered = list(paths)
    # Representative sampling: the raw DOTA val list is ordered small-images-first, so a
    # leading slice under-samples large scenes (fewer tiles) and understates sliced
    # latency. Draw a fixed-seed random subset that spans the image-size distribution.
    if len(ordered) > limit:
        ordered = random.Random(67).sample(ordered, limit)
    images: list[np.ndarray] = []
    for path in ordered:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
    if not images:
        raise FileNotFoundError("No benchmark images could be loaded.")
    return images


def time_pipeline(
    process_one: Callable[[np.ndarray], Any],
    images: Sequence[np.ndarray],
    warmup: int,
    iterations: int,
) -> list[float]:
    for img in images[: max(1, min(warmup, len(images)))]:
        process_one(img)
    sequence = [images[i % len(images)] for i in range(iterations)]
    times: list[float] = []
    for img in sequence:
        start = time.perf_counter()
        process_one(img)
        times.append((time.perf_counter() - start) * 1000.0)
    return times


def benchmark_native(
    model: Any,
    image_paths: Sequence[str | Path],
    imgsz: int,
    conf: float = 0.25,
    iou: float = 0.7,
    device: Any = None,
    half: bool = False,
    image_count: int = 25,
    warmup: int = 10,
    iterations: int = 100,
    max_det: int = 10000,
) -> dict[str, Any]:
    images = _load_images(image_paths, image_count)
    process = lambda img: native_detections(model, img, imgsz, conf, iou, device, half, max_det=max_det)
    times = time_pipeline(process, images, warmup, iterations)
    return {
        "mode": "native", "imgsz": imgsz, "tiles_per_image": 1.0,
        "image_count": len(images), "warmup": warmup, **summarize_latency(times),
    }


def benchmark_sliced(
    model: Any,
    image_paths: Sequence[str | Path],
    policy: SlicingPolicy,
    tile_imgsz: int | None = None,
    conf: float = 0.25,
    iou: float = 0.7,
    device: Any = None,
    half: bool = False,
    batch: int = 8,
    image_count: int = 25,
    warmup: int = 10,
    iterations: int = 100,
    max_det: int = 10000,
) -> dict[str, Any]:
    images = _load_images(image_paths, image_count)
    predictor = UltralyticsTilePredictor(
        model, imgsz=tile_imgsz or policy.tile_size, conf=conf, iou=iou,
        batch_size=batch, device=device, half=half, max_det=max_det,
    )
    process = lambda img: sliced_detections(img, policy, predictor)
    times = time_pipeline(process, images, warmup, iterations)
    mean_tiles = statistics.mean(len(windows_for_image(im.shape[1], im.shape[0], policy)) for im in images)
    return {
        "mode": "sliced", "policy": policy.label, "tile_size": policy.tile_size,
        "overlap": policy.overlap, "merge": policy.merge_method,
        "tiles_per_image": mean_tiles, "image_count": len(images), "warmup": warmup,
        **summarize_latency(times),
    }
