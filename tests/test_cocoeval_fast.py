"""The vectorised per-image matcher must agree with pycocotools exactly.

The paired image-level bootstrap reads per-image match arrays, which faster-coco-eval never
produces, so ``cocoeval_fast`` reimplements pycocotools' ``evaluateImg`` with the
ground-truth scan vectorised. These cases cover the rules that are easy to get wrong:
crowd ground truth absorbing several detections, instances ignored by the area range,
ties in IoU, and detections that match nothing.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import sys

import numpy as np
import pytest

pytest.importorskip("pycocotools")

from src.eval.cocoeval_fast import assert_matches_pycocotools, prepare_ious

_PYCOCOTOOLS_MODULES = ("pycocotools", "pycocotools.coco", "pycocotools.cocoeval", "pycocotools.mask")


def _genuine_pycocotools():
    """Import the real pycocotools even after faster-coco-eval has patched sys.modules.

    ``faster_coco_eval.init_as_pycocotools()`` swaps the four ``pycocotools`` entries in
    ``sys.modules`` for its own, and any earlier test that scores something triggers it.
    These cases exist to compare against the original implementation, so they drop those
    entries, import the installed package, and put the patch back for whatever runs next.
    """
    saved = {name: sys.modules.pop(name, None) for name in _PYCOCOTOOLS_MODULES}
    try:
        coco_module = importlib.import_module("pycocotools.coco")
        cocoeval_module = importlib.import_module("pycocotools.cocoeval")
        if not cocoeval_module.COCOeval.__module__.startswith("pycocotools"):
            pytest.skip("could not obtain unpatched pycocotools")
        return coco_module.COCO, cocoeval_module.COCOeval
    finally:
        for name, module in saved.items():
            if module is not None:
                sys.modules[name] = module


def _coco_eval(gt_dict, dt_records, area_rng):
    COCO, COCOeval = _genuine_pycocotools()
    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
        ev = COCOeval(coco_gt, coco_gt.loadRes(list(dt_records)), "bbox")
    ev.params.imgIds = sorted(coco_gt.getImgIds())
    ev.params.catIds = sorted(coco_gt.getCatIds())
    ev.params.maxDets = [100]
    ev.params.areaRng = area_rng
    ev.params.areaRngLbl = ["all"] * len(area_rng)
    with contextlib.redirect_stdout(io.StringIO()):
        prepare_ious(ev)
    return ev


def _gt(annotations):
    return {
        "images": [{"id": i, "file_name": str(i), "width": 200, "height": 200} for i in (0, 1)],
        "annotations": annotations,
        "categories": [{"id": 0, "name": "vehicle"}],
    }


def _ann(ann_id, image_id, bbox, iscrowd=0):
    return {"id": ann_id, "image_id": image_id, "category_id": 0, "bbox": list(bbox),
            "area": float(bbox[2] * bbox[3]), "iscrowd": iscrowd}


def _check(gt_dict, dt_records, area_rng):
    ev = _coco_eval(gt_dict, dt_records, area_rng)
    cells = [(i, c, a) for i in ev.params.imgIds for c in ev.params.catIds for a in area_rng]
    assert assert_matches_pycocotools(ev, cells, max_det=100) == len(cells)


ALL_AND_SMALL = [[0.0, 1e10], [0.0, 32.0 ** 2]]


def test_plain_matches_and_misses():
    gt_dict = _gt([_ann(1, 0, (10, 10, 20, 20)), _ann(2, 0, (100, 100, 10, 10)),
                   _ann(3, 1, (0, 0, 40, 40))])
    dt = [
        {"image_id": 0, "category_id": 0, "bbox": [10, 10, 20, 20], "score": 0.9},
        {"image_id": 0, "category_id": 0, "bbox": [12, 12, 20, 20], "score": 0.8},   # duplicate
        {"image_id": 0, "category_id": 0, "bbox": [150, 150, 10, 10], "score": 0.7},  # nothing
        {"image_id": 1, "category_id": 0, "bbox": [1, 1, 39, 39], "score": 0.95},
    ]
    _check(gt_dict, dt, ALL_AND_SMALL)


def test_crowd_ground_truth_absorbs_several_detections():
    gt_dict = _gt([_ann(1, 0, (10, 10, 60, 60), iscrowd=1), _ann(2, 1, (10, 10, 20, 20))])
    dt = [
        {"image_id": 0, "category_id": 0, "bbox": [10, 10, 60, 60], "score": 0.9},
        {"image_id": 0, "category_id": 0, "bbox": [11, 11, 59, 59], "score": 0.85},
        {"image_id": 0, "category_id": 0, "bbox": [12, 12, 58, 58], "score": 0.8},
        {"image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20], "score": 0.7},
    ]
    _check(gt_dict, dt, ALL_AND_SMALL)


def test_area_range_ignores_take_second_place():
    # A large instance is ignored under the small area range, so a detection over it may
    # only match it once nothing non-ignored is available.
    gt_dict = _gt([_ann(1, 0, (10, 10, 60, 60)), _ann(2, 0, (12, 12, 8, 8)),
                   _ann(3, 1, (100, 100, 60, 60))])
    dt = [
        {"image_id": 0, "category_id": 0, "bbox": [10, 10, 60, 60], "score": 0.9},
        {"image_id": 0, "category_id": 0, "bbox": [12, 12, 8, 8], "score": 0.8},
        {"image_id": 1, "category_id": 0, "bbox": [100, 100, 60, 60], "score": 0.6},
    ]
    _check(gt_dict, dt, ALL_AND_SMALL)


def test_tied_iou_resolves_the_same_way():
    # Two identical ground-truth boxes give a detection exactly equal IoU to both;
    # pycocotools keeps the highest index because it updates its best on ">=".
    gt_dict = _gt([_ann(1, 0, (20, 20, 20, 20)), _ann(2, 0, (20, 20, 20, 20)),
                   _ann(3, 1, (5, 5, 10, 10))])
    dt = [
        {"image_id": 0, "category_id": 0, "bbox": [20, 20, 20, 20], "score": 0.9},
        {"image_id": 0, "category_id": 0, "bbox": [20, 20, 20, 20], "score": 0.85},
        {"image_id": 1, "category_id": 0, "bbox": [5, 5, 10, 10], "score": 0.5},
    ]
    _check(gt_dict, dt, ALL_AND_SMALL)


def test_empty_cell_returns_none_like_pycocotools():
    gt_dict = _gt([_ann(1, 0, (10, 10, 20, 20))])
    dt = [{"image_id": 0, "category_id": 0, "bbox": [10, 10, 20, 20], "score": 0.9}]
    # image 1 has neither ground truth nor detections
    _check(gt_dict, dt, ALL_AND_SMALL)
