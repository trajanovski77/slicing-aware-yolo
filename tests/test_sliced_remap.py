"""Coordinate round-trip: slice -> per-tile detect -> offset -> merge recovers boxes.

Uses a fake detector that finds white blobs in each tile crop (one Detection per
connected component), so the test exercises the real tile_windows + offset + merge path
without torch/ultralytics.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.inference.merge import Detection, box_iou
from src.inference.sliced import SlicingPolicy, sliced_detections


def _blob_detector(crops):
    """Return one local Detection per white blob in each crop."""
    results = []
    for crop in crops:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h < 9:
                continue
            dets.append(Detection(0, 0.9, (float(x), float(y), float(x + w), float(y + h))))
        results.append(dets)
    return results


def _best_iou(box, boxes):
    return max((box_iou(box, b) for b in boxes), default=0.0)


def test_single_and_overlap_objects_recovered():
    image = np.zeros((800, 1000, 3), dtype=np.uint8)
    # A: fully inside a single tile. B: inside the x-overlap of two tiles (dedup test).
    box_a = (100, 100, 140, 140)
    box_b = (320, 100, 360, 140)
    for (x1, y1, x2, y2) in (box_a, box_b):
        cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), -1)

    policy = SlicingPolicy(tile_size=400, overlap=0.25, merge_method="nms_class_aware", merge_iou=0.55)
    result = sliced_detections(image, policy, _blob_detector)

    boxes = [d.xyxy for d in result.detections]
    # Exactly the two objects survive (B's duplicate across two tiles was merged).
    assert result.merged_detection_count == 2, boxes
    assert _best_iou(box_a, boxes) > 0.95
    assert _best_iou(box_b, boxes) > 0.95
    # B genuinely appeared in >1 tile pre-merge (raw > merged).
    assert result.raw_detection_count > result.merged_detection_count
    assert result.n_tiles >= 4


def test_offset_is_applied():
    """A blob far from the origin must come back in global (not tile-local) coords."""
    image = np.zeros((600, 600, 3), dtype=np.uint8)
    cv2.rectangle(image, (500, 500), (539, 539), (255, 255, 255), -1)
    policy = SlicingPolicy(tile_size=300, overlap=0.0, merge_method="nms_global", merge_iou=0.5)
    result = sliced_detections(image, policy, _blob_detector)
    assert result.detections, "expected at least one detection"
    assert _best_iou((500, 500, 540, 540), [d.xyxy for d in result.detections]) > 0.9
