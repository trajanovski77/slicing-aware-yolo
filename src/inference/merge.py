"""Box-merge primitives for slicing-aware inference.

Lifted (self-contained) from the prior repo's ``src/experiments/ensemble.py`` so this
repo carries no dependency on the ensemble/evaluator stack. ``nms_fusion`` is the
class-aware greedy NMS referenced as "class-aware NMS" in the proposal; ``nms_global``
is its class-agnostic counterpart; ``weighted_box_fusion`` (WBF) is an alternative
merge kept for the merge-strategy ablation (RQ5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    xyxy: tuple[float, float, float, float]
    source: str = ""
    weight: float = 1.0


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def nms_fusion(detections: list[Detection], iou_threshold: float = 0.5) -> list[Detection]:
    """Class-aware greedy NMS: suppress only same-class overlaps."""
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if all(
            detection.class_id != item.class_id or box_iou(detection.xyxy, item.xyxy) < iou_threshold
            for item in kept
        ):
            kept.append(detection)
    return kept


def nms_global(detections: list[Detection], iou_threshold: float = 0.5) -> list[Detection]:
    """Class-agnostic greedy NMS: suppress overlaps regardless of class."""
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if all(box_iou(detection.xyxy, item.xyxy) < iou_threshold for item in kept):
            kept.append(detection)
    return kept


def weighted_box_fusion(detections: list[Detection], iou_threshold: float = 0.5) -> list[Detection]:
    """Class-aware weighted box fusion — averages overlapping boxes instead of dropping."""
    remaining = sorted(detections, key=lambda item: item.confidence, reverse=True)
    fused: list[Detection] = []
    while remaining:
        anchor = remaining.pop(0)
        group = [anchor]
        rest: list[Detection] = []
        for detection in remaining:
            if detection.class_id == anchor.class_id and box_iou(detection.xyxy, anchor.xyxy) >= iou_threshold:
                group.append(detection)
            else:
                rest.append(detection)
        remaining = rest
        weights = [max(1e-9, item.confidence * item.weight) for item in group]
        total = sum(weights)
        coords = tuple(
            sum(item.xyxy[idx] * weight for item, weight in zip(group, weights)) / total
            for idx in range(4)
        )
        confidence = max(item.confidence for item in group)
        fused.append(
            Detection(
                class_id=anchor.class_id,
                confidence=confidence,
                xyxy=coords,  # type: ignore[arg-type]
                source="+".join(sorted({item.source for item in group if item.source})),
                weight=sum(item.weight for item in group) / len(group),
            )
        )
    return fused


_MERGERS: dict[str, Callable[[list[Detection], float], list[Detection]]] = {
    "nms_class_aware": nms_fusion,
    "nms_global": nms_global,
    "wbf": weighted_box_fusion,
}


def merge_detections(
    detections: list[Detection], method: str = "nms_class_aware", iou_threshold: float = 0.55
) -> list[Detection]:
    """Dispatch to the named merge strategy. Raises on an unknown method."""
    try:
        merger = _MERGERS[method]
    except KeyError as exc:
        raise ValueError(
            f"Unknown merge method '{method}'. Options: {sorted(_MERGERS)}"
        ) from exc
    return merger(detections, iou_threshold)


def detections_from_ultralytics(
    result: Any, offset: tuple[float, float] = (0.0, 0.0), source: str = ""
) -> list[Detection]:
    """Pull boxes off an Ultralytics result and shift them by ``offset`` (tile origin).

    Mirrors ``ensemble._detections_from_result`` but adds the tile-origin offset that
    remaps a tile's local coordinates back to the original image.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        boxes = getattr(result, "obb", None)
    if boxes is None or getattr(boxes, "xyxy", None) is None:
        return []
    ox, oy = offset
    xyxy = boxes.xyxy.cpu().tolist()
    conf = boxes.conf.cpu().tolist()
    cls = boxes.cls.cpu().tolist()
    detections: list[Detection] = []
    for box, score, class_id in zip(xyxy, conf, cls):
        x1, y1, x2, y2 = (float(v) for v in box)
        detections.append(
            Detection(
                class_id=int(class_id),
                confidence=float(score),
                xyxy=(x1 + ox, y1 + oy, x2 + ox, y2 + oy),
                source=source,
            )
        )
    return detections
