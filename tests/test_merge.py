from __future__ import annotations

import pytest

from src.inference.merge import (
    Detection, merge_detections, nms_fusion, nms_global, weighted_box_fusion,
)


def _d(cls, conf, box):
    return Detection(cls, conf, box)


def test_class_aware_nms_keeps_other_classes():
    dets = [
        _d(0, 0.9, (0, 0, 10, 10)),
        _d(0, 0.8, (1, 1, 11, 11)),   # same class, high overlap -> suppressed
        _d(1, 0.7, (0, 0, 10, 10)),   # different class, same box -> kept
    ]
    kept = nms_fusion(dets, iou_threshold=0.5)
    assert len(kept) == 2
    assert {d.class_id for d in kept} == {0, 1}
    # the higher-confidence class-0 box survives
    assert max(d.confidence for d in kept if d.class_id == 0) == 0.9


def test_global_nms_suppresses_across_classes():
    dets = [_d(0, 0.9, (0, 0, 10, 10)), _d(1, 0.8, (1, 1, 11, 11))]
    assert len(nms_global(dets, 0.5)) == 1
    assert len(nms_fusion(dets, 0.5)) == 2  # class-aware keeps both


def test_wbf_averages_boxes():
    dets = [_d(0, 0.9, (0, 0, 10, 10)), _d(0, 0.9, (2, 2, 12, 12))]
    fused = weighted_box_fusion(dets, 0.3)
    assert len(fused) == 1
    x1, y1, x2, y2 = fused[0].xyxy
    assert x1 == pytest.approx(1.0, abs=0.01)   # averaged origin
    assert x2 == pytest.approx(11.0, abs=0.01)
    assert fused[0].confidence == pytest.approx(0.9)


def test_merge_dispatch_and_unknown():
    dets = [_d(0, 0.9, (0, 0, 10, 10)), _d(0, 0.8, (1, 1, 11, 11))]
    assert len(merge_detections(dets, "nms_class_aware", 0.5)) == 1
    with pytest.raises(ValueError):
        merge_detections(dets, "does_not_exist", 0.5)


def test_non_overlapping_all_kept():
    dets = [_d(0, 0.9, (0, 0, 10, 10)), _d(0, 0.8, (100, 100, 110, 110))]
    assert len(nms_fusion(dets, 0.5)) == 2
