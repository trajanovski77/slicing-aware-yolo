"""Slicing-specific diagnostics (proposal RQ5).

- **Duplicate-box rate**: fraction of pre-merge detections removed by NMS/merge — the cost
  of overlapping tiles. Uses the pre/post counts that ``SlicedResult`` carries.
- **Boundary false positives**: detections whose centre sits near an *interior* tile seam
  and that match no GT box — the classic tile-edge artefact.
- **Detections per image** before/after merge — a calibration/health signal.

Functions operate on already-collected data (no re-inference) so they are unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from src.eval.matching import greedy_match
from src.inference.merge import Detection
from src.inference.sliced import SlicingPolicy, windows_for_image


@dataclass(frozen=True)
class ImageMergeCounts:
    stem: str
    raw: int      # detections before merge
    merged: int   # detections after merge
    n_tiles: int


def duplicate_rate(counts: Iterable[ImageMergeCounts]) -> dict[str, float]:
    counts = list(counts)
    raw = sum(c.raw for c in counts)
    merged = sum(c.merged for c in counts)
    tiles = sum(c.n_tiles for c in counts)
    n = max(1, len(counts))
    return {
        "duplicate_box_rate": (raw - merged) / raw if raw else 0.0,
        "raw_detections": raw,
        "merged_detections": merged,
        "mean_tiles_per_image": tiles / n,
        "mean_dets_per_image_raw": raw / n,
        "mean_dets_per_image_merged": merged / n,
        "images": len(counts),
    }


def interior_seams(width: int, height: int, policy: SlicingPolicy) -> tuple[list[int], list[int]]:
    """X and Y coordinates of interior tile edges (excluding the image border)."""
    windows = windows_for_image(width, height, policy)
    xs = {w.x for w in windows if w.x > 0} | {w.x2 for w in windows if w.x2 < width}
    ys = {w.y for w in windows if w.y > 0} | {w.y2 for w in windows if w.y2 < height}
    return sorted(xs), sorted(ys)


def _near_seam(det: Detection, seams: tuple[list[int], list[int]], margin: float) -> bool:
    x1, y1, x2, y2 = det.xyxy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    xs, ys = seams
    return any(abs(cx - s) <= margin for s in xs) or any(abs(cy - s) <= margin for s in ys)


def boundary_false_positives(
    detections: Sequence[Detection],
    gt_boxes: Sequence[tuple[int, tuple[float, float, float, float]]],
    width: int,
    height: int,
    policy: SlicingPolicy,
    iou_threshold: float = 0.5,
    margin: float | None = None,
) -> dict[str, int]:
    """Per-image boundary-FP counts. ``gt_boxes`` are ``(class_id, xyxy)`` in image coords."""
    margin = float(policy.overlap_px) if margin is None else margin
    seams = interior_seams(width, height, policy)
    dets = list(detections)
    if not dets:
        return {"near_seam_fp": 0, "total_fp": 0, "near_seam_detections": 0, "detections": 0}
    boxes = np.asarray([d.xyxy for d in dets], dtype=np.float64).reshape(len(dets), 4)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    xs = np.asarray(seams[0], dtype=np.float64)
    ys = np.asarray(seams[1], dtype=np.float64)
    near_x = (np.abs(cx[:, None] - xs[None, :]) <= margin).any(axis=1) if xs.size else np.zeros(len(dets), bool)
    near_y = (np.abs(cy[:, None] - ys[None, :]) <= margin).any(axis=1) if ys.size else np.zeros(len(dets), bool)
    near = near_x | near_y
    matched = greedy_match(
        [d.class_id for d in dets], [d.confidence for d in dets], boxes,
        [c for c, _ in gt_boxes], [b for _, b in gt_boxes], iou_threshold,
    )
    fp_mask = ~matched
    return {
        "near_seam_fp": int((fp_mask & near).sum()),
        "total_fp": int(fp_mask.sum()),
        "near_seam_detections": int(near.sum()),
        "detections": len(dets),
    }


def aggregate_boundary(per_image: Iterable[dict[str, int]]) -> dict[str, float]:
    rows = list(per_image)
    near_fp = sum(r["near_seam_fp"] for r in rows)
    total_fp = sum(r["total_fp"] for r in rows)
    near_total = sum(r["near_seam_detections"] for r in rows)
    dets = sum(r["detections"] for r in rows)
    return {
        "boundary_fp_rate_of_fps": near_fp / total_fp if total_fp else 0.0,
        "boundary_fp_rate_of_dets": near_fp / dets if dets else 0.0,
        "near_seam_fraction": near_total / dets if dets else 0.0,
        "near_seam_fp": near_fp,
        "total_fp": total_fp,
    }
