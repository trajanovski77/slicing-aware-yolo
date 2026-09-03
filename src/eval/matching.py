"""Vectorised greedy IoU matching shared by ``pr_f1`` and the seam diagnostics.

Semantics are identical to the original pure-Python loop: detections are processed in
descending score order (stable), each one claims the best still-unclaimed ground-truth box
of its own class, and it counts as matched when that IoU reaches ``iou_threshold``. The
NumPy version processes one class at a time (classes never compete for boxes, so the order
across classes does not matter) and replaces the inner Python loop over ground-truth boxes
with one IoU row per detection, which turns minutes per dense DOTA scene into milliseconds.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of xyxy boxes ``a`` (N,4) and ``b`` (M,4), same arithmetic as box_iou."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def greedy_match(
    det_classes: Sequence[int],
    det_scores: Sequence[float],
    det_boxes: Sequence[Sequence[float]],
    gt_classes: Sequence[int],
    gt_boxes: Sequence[Sequence[float]],
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Return a boolean array, one entry per detection (input order), True when matched."""
    n = len(det_scores)
    matched = np.zeros(n, dtype=bool)
    if n == 0 or len(gt_classes) == 0:
        return matched
    dc = np.asarray(det_classes, dtype=np.int64)
    ds = np.asarray(det_scores, dtype=np.float64)
    db = np.asarray(det_boxes, dtype=np.float64).reshape(n, 4)
    gc = np.asarray(gt_classes, dtype=np.int64)
    gb = np.asarray(gt_boxes, dtype=np.float64).reshape(len(gt_classes), 4)
    order = np.argsort(-ds, kind="stable")          # highest score first, ties keep input order
    for cls in np.unique(dc):
        d_idx = order[dc[order] == cls]
        g_idx = np.flatnonzero(gc == cls)
        if g_idx.size == 0:
            continue
        ious = iou_matrix(db[d_idx], gb[g_idx])
        claimed = np.zeros(g_idx.size, dtype=bool)
        for r, di in enumerate(d_idx):
            row = ious[r]
            if claimed.any():
                row = np.where(claimed, -1.0, row)
            j = int(np.argmax(row))
            if row[j] >= iou_threshold and row[j] > 0.0:
                claimed[j] = True
                matched[di] = True
    return matched
