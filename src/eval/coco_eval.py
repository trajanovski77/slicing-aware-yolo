"""COCO evaluation shared by native and sliced predictions (pycocotools).

Answers RQ1 (mAP@50, mAP@[50:95]) and RQ2 (AP/AR for small/medium/large via COCO's
absolute-pixel ``areaRng``) with one evaluator, plus per-category AP. ``pr_f1`` adds the
precision/recall/F1 at the operating confidence that the proposal's Table 1 asks for
(COCO's summary integrates over confidence, so P/R at a fixed operating point is computed
separately with a greedy IoU matcher).
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any, Mapping

# COCO summary stat indices (bbox, default areaRng; maxDets = [1, 10, DEFAULT_MAX_DETS]).
# pycocotools' stock maxDets=[1,10,100] keeps only the 100 highest-scoring detections per
# image-category cell. DOTA scenes routinely hold several hundred instances of one class,
# so the default silently truncates both native and sliced predictions (and caps recall at
# 0.405 on DOTA-v1.5 val). We therefore evaluate with a cap far above the densest cell.
DEFAULT_MAX_DETS = 10000
_STAT_KEYS = [
    "map50_95", "map50", "map75",
    "ap_small", "ap_medium", "ap_large",
    "ar_1", "ar_10", "ar_max",
    "ar_small", "ar_medium", "ar_large",
]


def _require_pycocotools():
    # faster-coco-eval is a C++ drop-in that reproduces pycocotools bit-for-bit and makes
    # high maxDets tractable (stock pycocotools' per-image matching loop is O(T*D*G) in
    # pure Python). Use it when installed; fall back to pycocotools otherwise.
    try:
        import faster_coco_eval

        faster_coco_eval.init_as_pycocotools()
    except ImportError:
        pass
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:  # pragma: no cover - env dependent
        raise ImportError(
            "pycocotools is required for evaluation. Install with `pip install pycocotools`."
        ) from exc
    return COCO, COCOeval


def _coco_from_dict(gt: Mapping[str, Any]):
    COCO, _ = _require_pycocotools()
    coco = COCO()
    coco.dataset = dict(gt)
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def run_coco_eval(
    gt: Mapping[str, Any] | str | Path,
    dt_records: list[dict[str, Any]] | str | Path,
    max_dets: int = DEFAULT_MAX_DETS,
) -> dict[str, Any]:
    """Return the 12 COCO summary stats plus per-category AP@[50:95] and AP@50.

    ``max_dets`` replaces COCO's per-image-category cap of 100 (params.maxDets[-1]).
    Passing 100 reproduces the stock pycocotools numbers exactly.
    """
    COCO, COCOeval = _require_pycocotools()
    if isinstance(gt, (str, Path)):
        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt = COCO(str(gt))
    else:
        coco_gt = _coco_from_dict(gt)

    # A variant that produced no detections scores zero on every metric (don't crash).
    if not isinstance(dt_records, (str, Path)) and not dt_records:
        cats = coco_gt.dataset.get("categories", [])
        return {
            **{key: 0.0 for key in _STAT_KEYS},
            "per_class": [
                {"class_id": int(c["id"]), "class_name": c["name"], "ap50_95": 0.0, "ap50": 0.0}
                for c in cats
            ],
        }

    coco_dt = coco_gt.loadRes(str(dt_records)) if isinstance(dt_records, (str, Path)) else coco_gt.loadRes(dt_records)

    with contextlib.redirect_stdout(io.StringIO()):
        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        evaluator.params.maxDets = [1, 10, int(max_dets)]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    stats = list(evaluator.stats)
    # summarize() hard-codes maxDets=100 for stats[0] (AP@[50:95]) and returns -1 when 100
    # is not in params.maxDets; recompute it at the largest cap from the precision tensor
    # exactly as _summarize does (mean over IoU thresholds and recall points, area=all).
    pr_all = evaluator.eval["precision"][:, :, :, 0, -1]
    stats[0] = float(pr_all[pr_all > -1].mean()) if (pr_all > -1).any() else -1.0
    out: dict[str, Any] = {key: float(stats[idx]) for idx, key in enumerate(_STAT_KEYS)}
    out["max_dets"] = int(max_dets)
    out["per_class"] = _per_category_ap(evaluator, coco_gt)
    return out


def _per_category_ap(evaluator: Any, coco_gt: Any) -> list[dict[str, Any]]:
    """Extract per-category AP from the precision tensor (single eval pass).

    precision shape: [T(iou), R(recall), K(cat), A(area), M(maxDet)].
    AP@[50:95] = mean over T,R at area=all, maxDet=params.maxDets[-1]; AP@50 = mean over R at T=0.
    """
    precision = evaluator.eval["precision"]  # numpy array
    cat_ids = evaluator.params.catIds
    id_to_name = {c["id"]: c["name"] for c in coco_gt.dataset.get("categories", [])}
    rows: list[dict[str, Any]] = []
    for k, cat_id in enumerate(cat_ids):
        pr_all = precision[:, :, k, 0, -1]
        pr_50 = precision[0, :, k, 0, -1]
        ap_all = float(pr_all[pr_all > -1].mean()) if (pr_all > -1).any() else float("nan")
        ap_50 = float(pr_50[pr_50 > -1].mean()) if (pr_50 > -1).any() else float("nan")
        rows.append({
            "class_id": int(cat_id),
            "class_name": id_to_name.get(cat_id, str(cat_id)),
            "ap50_95": ap_all,
            "ap50": ap_50,
        })
    return rows


def pr_f1(
    gt: Mapping[str, Any],
    dt_records: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Micro precision/recall/F1 at the operating confidence via greedy IoU matching.

    Detections are assumed already thresholded at the inference ``conf``. Per image, match
    each detection (highest score first) to the best unclaimed same-class GT box.
    """
    from src.eval.matching import greedy_match

    gt_by_image: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = {}
    for ann in gt["annotations"]:
        x, y, w, h = ann["bbox"]
        gt_by_image.setdefault(int(ann["image_id"]), []).append(
            (int(ann["category_id"]), (x, y, x + w, y + h))
        )
    dt_by_image: dict[int, list[dict[str, Any]]] = {}
    for rec in dt_records:
        dt_by_image.setdefault(int(rec["image_id"]), []).append(rec)

    tp = fp = 0
    total_gt = sum(len(v) for v in gt_by_image.values())
    for image_id, dets in dt_by_image.items():
        gts = gt_by_image.get(image_id, [])
        matched = greedy_match(
            [int(r["category_id"]) for r in dets],
            [float(r["score"]) for r in dets],
            [(r["bbox"][0], r["bbox"][1], r["bbox"][0] + r["bbox"][2], r["bbox"][1] + r["bbox"][3]) for r in dets],
            [c for c, _ in gts], [b for _, b in gts], iou_threshold,
        )
        n_tp = int(matched.sum())
        tp += n_tp
        fp += len(dets) - n_tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / total_gt if total_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "n_gt": total_gt}
