"""Image-level interleaved latency sweep for native + sliced variants (batch 1).

The sequential harness in ``sliced_speed`` times one (model, variant) pair to completion
before moving to the next. On a card shared with another job that ordering is unsafe: a
competing load that rises or falls during the session lands entirely on whichever pairs
happen to run inside it, so the block that ran first can look several times slower than an
identical block that ran later. The 2026-08-29 DOTA sweep shows exactly that, with the
three model blocks degrading in execution order (YOLOv8m 21m42s, YOLO11m 16m49s, YOLO26m
11m29s) and YOLOv8m returning a tile-640 median 2.8 times YOLO11m's for the same policy.

This driver instead cycles every (model, variant) pair once per timed image, rotating the
order each round so no pair keeps a fixed position in the cycle. Drift in the competing
load is then shared by all pairs rather than attributed to one, which keeps the relative
comparisons the paper makes (variant against variant, model against model) usable. It
cannot make the absolute milliseconds trustworthy: contention inflates every measurement,
so a sweep taken on a busy card is provisional and a quiet-card sweep remains the
authority. ``sample_gpu_load`` records what the card was doing so the two can be told
apart after the fact.

Images are loaded once and shared by every pair, unlike the sequential harness which
re-reads them per pair.
"""

from __future__ import annotations

import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from src.experiments.sliced_speed import summarize_latency
from src.inference.native import native_detections
from src.inference.sliced import UltralyticsTilePredictor, sliced_detections, windows_for_image


@dataclass
class Runner:
    """One (model, variant) pair, ready to be timed on a single image."""

    model_name: str
    variant_label: str
    process: Callable[[np.ndarray], Any]
    meta: dict[str, Any]
    times_ms: list[float] = field(default_factory=list)


def build_runners(
    loaded_models: Sequence[tuple[dict[str, Any], Any]],
    variants: Sequence[Any],
    images: Sequence[np.ndarray],
    conf: float,
    iou: float,
    device: Any,
    half: bool,
    batch: int,
    max_det: int,
) -> list[Runner]:
    """One runner per (model, variant), with the tile predictors built up front."""
    runners: list[Runner] = []
    for model_cfg, model in loaded_models:
        for variant in variants:
            if variant.kind == "native":
                imgsz = int(variant.imgsz)
                process = (
                    lambda img, m=model, s=imgsz: native_detections(
                        m, img, s, conf, iou, device, half, max_det=max_det
                    )
                )
                meta = {"mode": "native", "imgsz": imgsz, "tiles_per_image": 1.0}
            else:
                policy = variant.policy
                predictor = UltralyticsTilePredictor(
                    model, imgsz=policy.tile_size, conf=conf, iou=iou,
                    batch_size=batch, device=device, half=half, max_det=max_det,
                )
                process = (
                    lambda img, p=policy, pr=predictor: sliced_detections(img, p, pr)
                )
                mean_tiles = statistics.mean(
                    len(windows_for_image(im.shape[1], im.shape[0], policy)) for im in images
                )
                meta = {
                    "mode": "sliced", "policy": policy.label, "tile_size": policy.tile_size,
                    "overlap": policy.overlap, "merge": policy.merge_method,
                    "tiles_per_image": mean_tiles,
                }
            runners.append(Runner(model_cfg["name"], variant.label, process, meta))
    return runners


class GpuLoadSampler:
    """Poll one GPU's utilisation and memory in the background during a sweep.

    Recorded so a sweep taken on a shared card can be told from a quiet one later, rather
    than the reader having to trust that the card was idle.
    """

    def __init__(self, device: Any, interval_s: float = 2.0) -> None:
        self.index = self._device_index(device)
        self.interval_s = interval_s
        self._samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _device_index(device: Any) -> int | None:
        try:
            return int(str(device).strip())
        except (TypeError, ValueError):
            return None

    def _poll_once(self) -> tuple[float, float] | None:
        if self.index is None:
            return None
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits", "-i", str(self.index)],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout.strip()
            util, mem = (float(x) for x in out.split(","))
            return util, mem
        except (subprocess.SubprocessError, ValueError, OSError):
            return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample = self._poll_once()
            if sample is not None:
                self._samples.append(sample)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "GpuLoadSampler":
        if self.index is not None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 5)

    def summary(self) -> dict[str, Any]:
        if not self._samples:
            return {"samples": 0}
        util = np.array([s[0] for s in self._samples])
        mem = np.array([s[1] for s in self._samples])
        return {
            "samples": len(self._samples),
            "device": self.index,
            "util_median_pct": float(np.median(util)),
            "util_p10_pct": float(np.percentile(util, 10)),
            "util_max_pct": float(util.max()),
            "mem_used_median_mib": float(np.median(mem)),
            "mem_used_max_mib": float(mem.max()),
        }


def interleaved_sweep(
    runners: Sequence[Runner],
    images: Sequence[np.ndarray],
    warmup: int,
    iterations: int,
    on_round: Callable[[int, int], None] | None = None,
) -> None:
    """Time every runner once per image, rotating the cycle order each round.

    Fills ``runner.times_ms`` in place with one measurement per timed round.
    """
    for runner in runners:
        for img in images[: max(1, min(warmup, len(images)))]:
            runner.process(img)
    count = len(runners)
    for i in range(iterations):
        img = images[i % len(images)]
        offset = i % count
        for runner in list(runners[offset:]) + list(runners[:offset]):
            start = time.perf_counter()
            runner.process(img)
            runner.times_ms.append((time.perf_counter() - start) * 1000.0)
        if on_round is not None:
            on_round(i + 1, iterations)


def results_from(runners: Sequence[Runner], image_count: int, warmup: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for runner in runners:
        row = dict(runner.meta)
        row.update({
            "model": runner.model_name, "variant": runner.variant_label,
            "image_count": image_count, "warmup": warmup,
            **summarize_latency(runner.times_ms),
        })
        rows.append(row)
    return rows
