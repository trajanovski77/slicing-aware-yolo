"""Vectorised per-image COCO matching, for uses that need ``evalImgs`` and speed.

pycocotools' ``COCOeval.evaluateImg`` greedily matches detections to ground truth with a
Python loop nested three deep (IoU threshold, detection, ground truth). On DOTA that is
fatal: a dense sliced variant puts thousands of detections and hundreds of instances in one
image-category cell, and the innermost loop runs once per pair per threshold. The repo
normally sidesteps this with faster-coco-eval, whose C++ path is a drop-in for scoring but
deliberately never populates ``evalImgs`` and de-duplicates ``params.imgIds``, so anything
that needs per-image match arrays (the paired image-level bootstrap) cannot use it.

``evaluate_img_arrays`` reproduces ``evaluateImg`` exactly with the ground-truth scan
vectorised, leaving the greedy order over detections intact because it is inherently
sequential. The IoU matrices still come from pycocotools' own ``computeIoU``, which is
already C. ``assert_matches_pycocotools`` checks cell for cell against the original.

The matching rules being reproduced, in pycocotools' own order of precedence:

* ground truth is sorted with non-ignored instances first, detections by descending score;
* a detection may only match ground truth that is unmatched at this threshold, unless that
  ground truth is a crowd, which may absorb any number of detections;
* among admissible candidates at or above the threshold the highest IoU wins, and ties go
  to the highest index, because the original updates its running best on ``>=``;
* non-ignored ground truth is preferred outright: ignored instances are considered only
  when nothing non-ignored matched, which is what the original's ``break`` achieves;
* an unmatched detection whose own area falls outside the area range is ignored.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def prepare_ious(ev) -> None:
    """Fill ``ev._gts``/``ev._dts`` and ``ev.ious`` using pycocotools' own routines."""
    ev._prepare()
    cat_ids = ev.params.catIds if ev.params.useCats else [-1]
    ev.ious = {
        (img_id, cat_id): ev.computeIoU(img_id, cat_id)
        for img_id in ev.params.imgIds
        for cat_id in cat_ids
    }


def evaluate_img_arrays(ev, img_id: int, cat_id: int, area_rng, max_det: int) -> dict[str, Any] | None:
    """Same return value as ``COCOeval.evaluateImg``, with the ground-truth scan vectorised."""
    p = ev.params
    if p.useCats:
        gt, dt = ev._gts[img_id, cat_id], ev._dts[img_id, cat_id]
    else:
        gt = [g for c_id in p.catIds for g in ev._gts[img_id, c_id]]
        dt = [d for c_id in p.catIds for d in ev._dts[img_id, c_id]]
    if len(gt) == 0 and len(dt) == 0:
        return None

    for g in gt:
        g["_ignore"] = 1 if (g["ignore"] or g["area"] < area_rng[0] or g["area"] > area_rng[1]) else 0
    gt_order = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
    gt = [gt[i] for i in gt_order]
    dt_order = np.argsort([-d["score"] for d in dt], kind="mergesort")
    dt = [dt[i] for i in dt_order[:max_det]]

    is_crowd = np.array([int(g["iscrowd"]) for g in gt], dtype=bool)
    gt_ignore = np.array([g["_ignore"] for g in gt], dtype=np.int64)
    ious_all = ev.ious[img_id, cat_id]
    ious = ious_all[:, gt_order] if len(ious_all) > 0 else ious_all

    n_iou, n_gt, n_dt = len(p.iouThrs), len(gt), len(dt)
    dt_match = np.zeros((n_iou, n_dt))
    dt_ig = np.zeros((n_iou, n_dt))

    if len(ious) > 0 and n_dt > 0 and n_gt > 0:
        gt_ids = np.array([g["id"] for g in gt])
        not_ignored = gt_ignore == 0
        for t_idx, thr in enumerate(p.iouThrs):
            floor = min(thr, 1 - 1e-10)
            taken = np.zeros(n_gt, dtype=bool)
            for d_idx in range(n_dt):
                row = ious[d_idx]
                admissible = ((~taken) | is_crowd) & (row >= floor)
                pick = -1
                # Non-ignored ground truth first; ignored only if nothing else matched.
                for group in (admissible & not_ignored, admissible & ~not_ignored):
                    if group.any():
                        candidates = np.where(group, row, -1.0)
                        best = candidates.max()
                        pick = int(np.flatnonzero(candidates == best)[-1])
                        break
                if pick < 0:
                    continue
                dt_ig[t_idx, d_idx] = gt_ignore[pick]
                dt_match[t_idx, d_idx] = gt_ids[pick]
                if not is_crowd[pick]:
                    taken[pick] = True

    outside = np.array(
        [d["area"] < area_rng[0] or d["area"] > area_rng[1] for d in dt], dtype=bool
    ).reshape((1, n_dt))
    dt_ig = np.logical_or(dt_ig, np.logical_and(dt_match == 0, np.repeat(outside, n_iou, 0)))
    return {
        "image_id": img_id,
        "category_id": cat_id,
        "aRng": area_rng,
        "maxDet": max_det,
        "dtIds": [d["id"] for d in dt],
        "gtIds": [g["id"] for g in gt],
        "dtMatches": dt_match,
        "dtScores": [d["score"] for d in dt],
        "gtIgnore": gt_ignore,
        "dtIgnore": dt_ig,
    }


def assert_matches_pycocotools(ev, cells, max_det: int, tol: float = 0.0) -> int:
    """Compare against pycocotools' own ``evaluateImg`` on the given cells.

    ``cells`` is an iterable of ``(img_id, cat_id, area_rng)``. Returns the number of cells
    checked; raises on the first disagreement.
    """
    checked = 0
    for img_id, cat_id, area_rng in cells:
        ours = evaluate_img_arrays(ev, img_id, cat_id, area_rng, max_det)
        theirs = ev.evaluateImg(img_id, cat_id, area_rng, max_det)
        if ours is None or theirs is None:
            if ours is not theirs:
                raise RuntimeError(f"cell {(img_id, cat_id, tuple(area_rng))}: one is None, the other is not")
            checked += 1
            continue
        for key in ("dtMatches", "dtIgnore", "gtIgnore", "dtScores"):
            a = np.asarray(ours[key], dtype=float)
            b = np.asarray(theirs[key], dtype=float)
            if a.shape != b.shape or not np.allclose(a, b, atol=tol, rtol=0):
                raise RuntimeError(
                    f"cell {(img_id, cat_id, tuple(area_rng))}: {key} differs from pycocotools"
                )
        checked += 1
    return checked
