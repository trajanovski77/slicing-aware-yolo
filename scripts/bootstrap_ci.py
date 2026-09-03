#!/usr/bin/env python3
"""Paired image-level bootstrap CIs for the headline DOTA contrasts (F39).

Resamples the 458 validation scenes with replacement (one shared resample per replicate
for every variant, so the differences are paired), recomputes AP from the cached per-image
matches, and reports 95 % percentile intervals for mAP@[50:95], mAP@50 and AP_S plus their
differences against native1024.

    python scripts/bootstrap_ci.py --model yolo11m_dota --reps 300

Two things make this correct and tractable, both learned the hard way:

Backend. The matching must come from STOCK pycocotools. faster-coco-eval is a drop-in for
scoring but not for resampling: it never populates ``evalImgs`` (by design, as a
bottleneck), its ``accumulate()`` re-accumulates from a cached C++ structure and ignores
``params.imgIds`` entirely, and its ``evaluate()`` calls ``np.unique`` on ``imgIds`` so a
resample with duplicates silently collapses to the unique set. Under that backend every
replicate reproduces the point estimate and every interval comes out zero-width, which is
what the 2026-08-29 run produced. ``_stock_pycocotools`` refuses the patched backend and
``_assert_resampling_bites`` fails loudly if the replicates do not move.

Speed, in two places. The per-image matching comes from
``src.eval.cocoeval_fast.evaluate_img_arrays``, a vectorised equivalent of pycocotools'
``evaluateImg`` (whose triple Python loop is hopeless on a scene holding 8,609 instances),
checked cell for cell against the original. And stock ``accumulate()`` cannot be called 300
times, because it walks every detection in Python to build the monotone precision envelope;
``_accumulate_weighted`` does the same arithmetic in NumPy, expressing the resample as
per-image integer weights on a globally score-sorted array and the envelope as
``np.maximum.accumulate``. Both shortcuts are validated against stock pycocotools before
any replicate is drawn, so neither can change the numbers.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.coco_eval import DEFAULT_MAX_DETS  # noqa: E402
from src.eval.cocoeval_fast import (  # noqa: E402
    assert_matches_pycocotools, evaluate_img_arrays, prepare_ious,
)

VARIANTS = ["native1024", "native1536", "tile512_ov10", "tile640_ov10", "tile1024_ov10", "tile1024_ov20"]

# Only the "all" and "small" area ranges are needed (mAP, mAP@50, AP_S), so the other two
# are dropped before evaluate() to halve the per-image matching work.
AREA_RNG = [[0.0, 1e10], [0.0, 32.0 ** 2]]
AREA_LBL = ["all", "small"]


def _stock_pycocotools():
    """Return stock pycocotools' COCO/COCOeval, refusing a faster-coco-eval monkeypatch."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    for cls in (COCO, COCOeval):
        module = getattr(cls, "__module__", "")
        if not module.startswith("pycocotools"):
            raise RuntimeError(
                f"{cls.__name__} resolves to {module}, not pycocotools. The image-level "
                "resample needs stock pycocotools' per-image matches; a patched backend "
                "returns zero-width intervals. Do not call init_as_pycocotools() here."
            )
    return COCO, COCOeval


class VariantCells:
    """Per-(category, area) detection arrays for one variant, sorted by score.

    ``evalImgs`` from pycocotools holds, for every (category, area, image) cell, the
    per-detection match and ignore flags at each IoU threshold. AP for a resampled set of
    images is a function of those arrays and of how many times each image was drawn, so
    they are concatenated once here, sorted by descending score, and tagged with the image
    each detection came from. A replicate is then an integer weight vector over images.
    """

    def __init__(self, ev, n_iou: int, max_det: int) -> None:
        self.n_iou = n_iou
        self.n_images = len(list(ev.params.imgIds))
        n_cat, n_area = len(ev.params.catIds), len(ev.params.areaRng)
        self.cells: dict[tuple[int, int], dict] = {}
        for a in range(n_area):
            for k in range(n_cat):
                base = k * n_area * self.n_images + a * self.n_images
                scores, matches, ignores, owners = [], [], [], []
                npig_per_image = np.zeros(self.n_images)
                for i in range(self.n_images):
                    entry = ev.evalImgs[base + i]
                    if entry is None:
                        continue
                    gt_ignore = np.asarray(entry["gtIgnore"])
                    npig_per_image[i] = float(np.count_nonzero(gt_ignore == 0))
                    score = np.asarray(entry["dtScores"][:max_det], dtype=float)
                    if score.size == 0:
                        continue
                    scores.append(score)
                    matches.append(np.asarray(entry["dtMatches"])[:, :max_det])
                    ignores.append(np.asarray(entry["dtIgnore"])[:, :max_det])
                    owners.append(np.full(score.size, i, dtype=np.int32))
                if not scores:
                    self.cells[(k, a)] = {"npig_per_image": npig_per_image, "empty": True}
                    continue
                # mergesort to match pycocotools' stable ordering of equal scores.
                order = np.argsort(-np.concatenate(scores), kind="mergesort")
                dtm = np.concatenate(matches, axis=1)[:, order]
                dt_ig = np.concatenate(ignores, axis=1)[:, order].astype(bool)
                self.cells[(k, a)] = {
                    "npig_per_image": npig_per_image,
                    "empty": False,
                    "owner": np.concatenate(owners)[order],
                    "tp": np.logical_and(dtm, ~dt_ig),
                    "fp": np.logical_and(np.logical_not(dtm), ~dt_ig),
                }


def _accumulate_weighted(cells: VariantCells, weights: np.ndarray, rec_thrs: np.ndarray) -> tuple[float, float, float]:
    """AP over a weighted (resampled) image set: (mAP@[50:95], mAP@50, AP_S).

    Reproduces pycocotools' accumulate/summarize arithmetic. A detection from an image
    drawn w times contributes w to the running true/false positive counts, which is what
    concatenating that image's detections w times would do, and the number of positives
    scales the same way. Cells with no non-ignored ground truth are dropped, matching
    pycocotools leaving them at -1 and averaging only over entries above -1.
    """
    eps = np.spacing(1)
    n_rec = rec_thrs.size
    blocks: dict[int, list] = {0: [], 1: []}   # area index -> (n_iou, n_rec) precision blocks
    for (_, a), cell in cells.cells.items():
        npig = float(np.dot(weights, cell["npig_per_image"]))
        if npig <= 0:
            continue
        if cell["empty"]:
            blocks[a].append(np.zeros((cells.n_iou, n_rec)))
            continue
        w_det = weights[cell["owner"]].astype(float)
        tp_sum = np.cumsum(cell["tp"] * w_det, axis=1)
        fp_sum = np.cumsum(cell["fp"] * w_det, axis=1)
        recall = tp_sum / npig
        precision = tp_sum / (tp_sum + fp_sum + eps)
        # Monotone envelope, vectorised: pycocotools walks this backwards in Python.
        precision = np.maximum.accumulate(precision[:, ::-1], axis=1)[:, ::-1]
        block = np.zeros((cells.n_iou, n_rec))
        n_det = recall.shape[1]
        for t in range(cells.n_iou):
            idx = np.searchsorted(recall[t], rec_thrs, side="left")
            valid = idx < n_det
            block[t, valid] = precision[t, idx[valid]]
        blocks[a].append(block)

    def mean_over(area: int, iou_slice) -> float:
        if not blocks[area]:
            return float("nan")
        stacked = np.stack(blocks[area])        # (cells, n_iou, n_rec)
        return float(stacked[:, iou_slice, :].mean())

    return mean_over(0, slice(None)), mean_over(0, slice(0, 1)), mean_over(1, slice(None))


def _stats_from(ev) -> tuple[float, float, float]:
    """The same three statistics read out of a stock pycocotools precision tensor."""
    pr = ev.eval["precision"]

    def m(t=slice(None), a=0):
        x = pr[t, :, :, a, -1]
        return float(x[x > -1].mean()) if (x > -1).any() else float("nan")

    return m(), m(t=0), m(a=1)


def _assert_matches_pycocotools(cells: VariantCells, ev, variant: str, tol: float = 1e-6) -> None:
    """Check the vectorised accumulate against stock accumulate at unit weights."""
    with contextlib.redirect_stdout(io.StringIO()):
        ev.accumulate()
    reference = _stats_from(ev)
    ours = _accumulate_weighted(cells, np.ones(cells.n_images), ev.params.recThrs)
    for name, a, b in zip(("mAP", "mAP@50", "AP_S"), reference, ours):
        if not (abs(a - b) <= tol or (np.isnan(a) and np.isnan(b))):
            raise RuntimeError(
                f"{variant}: vectorised accumulate disagrees with pycocotools on {name} "
                f"({b:.9f} vs {a:.9f}); refusing to bootstrap from it."
            )


def _assert_resampling_bites(reps: dict, point: dict) -> None:
    """Fail if the replicates never move off the point estimate (a dead resample)."""
    for variant, rows in reps.items():
        if not (np.ptp(np.asarray(rows), axis=0) > 0).any():
            raise RuntimeError(
                f"bootstrap for {variant} produced identical statistics in all "
                f"{len(rows)} replicates (point {point[variant]}). The resample is not "
                "reaching the accumulate step; check the evaluator backend."
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11m_dota")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=67)
    ap.add_argument("--variants", default=None,
                    help="comma-separated subset of VARIANTS (native1024 is always kept as the paired baseline)")
    args = ap.parse_args()
    variants = VARIANTS
    if args.variants:
        wanted = [v.strip() for v in args.variants.split(",") if v.strip()]
        unknown = [v for v in wanted if v not in VARIANTS]
        if unknown:
            raise SystemExit(f"unknown variants: {unknown}; choose from {VARIANTS}")
        variants = ["native1024"] + [v for v in wanted if v != "native1024"]

    COCO, COCOeval = _stock_pycocotools()
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ROOT / "results" / "coco" / "dota_val_gt.json"))
    img_ids = sorted(gt.getImgIds())

    cells: dict[str, VariantCells] = {}
    point: dict[str, tuple[float, float, float]] = {}
    rec_thrs = None
    for variant in variants:
        started = time.time()
        dt = json.load(open(ROOT / "results" / "predictions" / "dota" / f"{args.model}__{variant}.json"))["dt"]
        with contextlib.redirect_stdout(io.StringIO()):
            ev = COCOeval(gt, gt.loadRes(dt), "bbox")
            ev.params.maxDets = [DEFAULT_MAX_DETS]
            ev.params.areaRng = AREA_RNG
            ev.params.areaRngLbl = AREA_LBL
            ev.params.imgIds = img_ids
            prepare_ious(ev)
        # Spot-check the vectorised matcher against pycocotools on this variant's own
        # data before trusting it, densest scene included.
        by_gt = sorted(img_ids, key=lambda i: -len(gt.getAnnIds(imgIds=[i])))
        probes = [by_gt[0], by_gt[len(by_gt) // 2], by_gt[-1]]
        checked = assert_matches_pycocotools(
            ev, [(i, c, a) for i in probes for c in ev.params.catIds[:4] for a in AREA_RNG],
            DEFAULT_MAX_DETS)
        # evalImgs in pycocotools' own (category, area, image) layout, so that stock
        # accumulate() can validate the weighted version below.
        ev.evalImgs = [
            evaluate_img_arrays(ev, i, c, a, DEFAULT_MAX_DETS)
            for c in ev.params.catIds for a in ev.params.areaRng for i in img_ids
        ]
        ev._paramsEval = copy.deepcopy(ev.params)
        packed = VariantCells(ev, len(ev.params.iouThrs), DEFAULT_MAX_DETS)
        _assert_matches_pycocotools(packed, ev, variant)
        cells[variant] = packed
        rec_thrs = ev.params.recThrs
        point[variant] = _accumulate_weighted(packed, np.ones(len(img_ids)), rec_thrs)
        print(f"{variant:14s} matcher and accumulate both agree with pycocotools "
              f"({checked} cells); mAP {point[variant][0]:.4f} AP_S {point[variant][2]:.4f} "
              f"({time.time() - started:.0f} s)", flush=True)
        del ev

    rng = np.random.default_rng(args.seed)
    n_images = len(img_ids)
    reps: dict[str, list] = {v: [] for v in variants}
    for r in range(args.reps):
        draw = rng.integers(0, n_images, size=n_images)
        weights = np.bincount(draw, minlength=n_images).astype(float)
        for variant in variants:
            reps[variant].append(_accumulate_weighted(cells[variant], weights, rec_thrs))
        if (r + 1) % 50 == 0:
            print(f"  {r + 1}/{args.reps} replicates", flush=True)
    _assert_resampling_bites(reps, point)

    keys = ("map", "map50", "aps")
    arr = {v: np.array(reps[v]) for v in variants}
    out = {
        "model": args.model, "reps": args.reps, "seed": args.seed,
        "variants": variants,
        "method": "paired image-level bootstrap; percentile intervals; stock pycocotools "
                  "matches with a vectorised weighted accumulate validated at unit weights",
        "point": {v: dict(zip(keys, point[v])) for v in variants},
        "ci": {}, "diff_vs_native1024": {},
    }
    for v in variants:
        lo, hi = np.percentile(arr[v], [2.5, 97.5], axis=0)
        out["ci"][v] = {k: [float(lo[i]), float(hi[i])] for i, k in enumerate(keys)}
        d = arr[v] - arr["native1024"]
        dlo, dhi = np.percentile(d, [2.5, 97.5], axis=0)
        out["diff_vs_native1024"][v] = {
            k: [float(point[v][i] - point["native1024"][i]), float(dlo[i]), float(dhi[i])]
            for i, k in enumerate(keys)
        }
    path = ROOT / "results" / "tables" / f"bootstrap_{args.model}.json"
    path.write_text(json.dumps(out, indent=1))
    for v in variants:
        p, c, d = out["point"][v], out["ci"][v], out["diff_vs_native1024"][v]
        print(f"{v:14s} mAP {p['map']:.3f} [{c['map'][0]:.3f},{c['map'][1]:.3f}]  "
              f"APs {p['aps']:.3f} [{c['aps'][0]:.3f},{c['aps'][1]:.3f}]  "
              f"dAPs vs native1024 {d['aps'][0]:+.3f} [{d['aps'][1]:+.3f},{d['aps'][2]:+.3f}]")
    print("->", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
