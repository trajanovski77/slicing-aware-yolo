"""Convert merged ``Detection`` lists into COCO detection (``dt``) records.

Image ids come from the COCO GT so ``dt`` and ``gt`` align; images are matched by stem,
which is robust to differing image suffixes between predictions and GT.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from src.inference.merge import Detection
from src.utils.paths import read_yaml, write_json


def image_id_index(coco_gt: Mapping[str, Any]) -> dict[str, int]:
    """Map image stem -> COCO image id from a loaded GT dict."""
    return {str(img["file_name"]): int(img["id"]) for img in coco_gt["images"]}


def detections_to_coco(
    per_image: Mapping[str, Iterable[Detection]],
    stem_to_id: Mapping[str, int],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for stem, detections in per_image.items():
        image_id = stem_to_id.get(stem)
        if image_id is None:
            continue
        for det in detections:
            x1, y1, x2, y2 = det.xyxy
            records.append({
                "image_id": image_id,
                "category_id": int(det.class_id),
                "bbox": [round(x1, 3), round(y1, 3), round(x2 - x1, 3), round(y2 - y1, 3)],
                "score": round(float(det.confidence), 6),
            })
    return records


def write_coco_dt(records: list[dict[str, Any]], output_json: str | Path) -> Path:
    write_json(records, output_json)
    return Path(output_json)


def load_coco_gt(path: str | Path) -> dict[str, Any]:
    import json

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _stem_to_id_from_yaml_or_gt(gt_path: str | Path) -> dict[str, int]:
    return image_id_index(load_coco_gt(gt_path))
