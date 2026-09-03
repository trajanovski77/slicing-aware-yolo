"""Build a COCO ground-truth JSON from a dataset's *original-resolution* val labels.

Two input modes, because native-vs-sliced must be scored on untiled originals:

- ``dota_raw``: original DOTA images + ``labelTxt`` (absolute polygon coords). Each OBB is
  reduced to its axis-aligned HBB (min/max of the four points), matching how the prior
  repo produced HBB detection labels.
- ``yolo``: an Ultralytics-format tree (``images/<split>`` + ``labels/<split>`` with
  normalised ``cls cx cy w h``). Covers ships / WHU / any already-YOLO dataset; box coords
  are denormalised with the image's own dimensions.

The same evaluator then scores native and every sliced policy, so the comparison is
apples-to-apples and independent of Ultralytics ``val``. ``file_name`` is the image *stem*
(no extension) so predictions can be matched back by stem regardless of image suffix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from src.datasets.converters import parse_dota_annotation_line
from src.datasets.dota import _find_split_dirs
from src.utils.paths import iter_images, read_yaml, write_json


def _names_list(dataset_yaml: str | Path) -> list[str]:
    names = read_yaml(dataset_yaml).get("names", {})
    if isinstance(names, list):
        return [str(n) for n in names]
    return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]


def _image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    height, width = image.shape[:2]
    return width, height


def _coco_skeleton(class_names: list[str]) -> dict[str, Any]:
    return {
        "images": [],
        "annotations": [],
        "categories": [{"id": idx, "name": name} for idx, name in enumerate(class_names)],
    }


def _add_hbb(coco: dict[str, Any], ann_id: int, image_id: int, class_id: int,
             x1: float, y1: float, x2: float, y2: float, difficult: int = 0) -> int:
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    if w <= 0 or h <= 0:
        return ann_id
    coco["annotations"].append({
        "id": ann_id,
        "image_id": image_id,
        "category_id": int(class_id),
        "bbox": [round(x1, 3), round(y1, 3), round(w, 3), round(h, 3)],
        "area": round(w * h, 3),
        "iscrowd": 0,
        "ignore": int(difficult),
    })
    return ann_id + 1


def build_from_dota_raw(
    raw_dir: str | Path,
    dataset_yaml: str | Path,
    split: str = "val",
    include_difficult: bool = True,
) -> dict[str, Any]:
    class_names = _names_list(dataset_yaml)
    image_dir, label_dir = _find_split_dirs(Path(raw_dir), split, task="detect")
    if image_dir is None or label_dir is None:
        raise FileNotFoundError(f"Could not locate DOTA {split} images/labels under {raw_dir}")
    coco = _coco_skeleton(class_names)
    ann_id = 1
    for image_id, image_path in enumerate(iter_images(image_dir)):
        width, height = _image_size(image_path)
        coco["images"].append({
            "id": image_id, "file_name": image_path.stem, "width": width, "height": height,
        })
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            ann = parse_dota_annotation_line(line, class_names)
            if ann is None:
                continue
            if ann.difficult and not include_difficult:
                continue
            xs = [p[0] for p in ann.points]
            ys = [p[1] for p in ann.points]
            ann_id = _add_hbb(
                coco, ann_id, image_id, ann.class_id,
                min(xs), min(ys), max(xs), max(ys), ann.difficult,
            )
    return coco


def build_from_yolo(
    images_dir: str | Path,
    labels_dir: str | Path,
    dataset_yaml: str | Path,
) -> dict[str, Any]:
    class_names = _names_list(dataset_yaml)
    coco = _coco_skeleton(class_names)
    labels_root = Path(labels_dir)
    ann_id = 1
    for image_id, image_path in enumerate(iter_images(images_dir)):
        width, height = _image_size(image_path)
        coco["images"].append({
            "id": image_id, "file_name": image_path.stem, "width": width, "height": height,
        })
        label_path = labels_root / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
            x1 = (cx - w / 2) * width
            y1 = (cy - h / 2) * height
            x2 = (cx + w / 2) * width
            y2 = (cy + h / 2) * height
            ann_id = _add_hbb(coco, ann_id, image_id, cls, x1, y1, x2, y2)
    return coco


def build_coco_gt(config: dict[str, Any]) -> dict[str, Any]:
    """Dispatch on ``gt_format`` in a slicing config: 'dota_raw' or 'yolo'."""
    fmt = str(config.get("gt_format", "dota_raw"))
    dataset_yaml = config["dataset"]
    split = str(config.get("split", "val"))
    if fmt == "dota_raw":
        return build_from_dota_raw(
            config["raw_dir"], dataset_yaml, split,
            include_difficult=bool(config.get("include_difficult", True)),
        )
    if fmt == "yolo":
        meta = read_yaml(dataset_yaml)
        root = Path(config.get("images_root", meta.get("path", ".")))
        images_dir = config.get("images_dir", root / meta.get(split, f"images/{split}"))
        labels_dir = config.get("labels_dir", root / f"labels/{split}")
        return build_from_yolo(images_dir, labels_dir, dataset_yaml)
    raise ValueError(f"Unknown gt_format '{fmt}' (expected 'dota_raw' or 'yolo')")


def write_coco_gt(coco: dict[str, Any], output_json: str | Path) -> Path:
    write_json(coco, output_json)
    return Path(output_json)
