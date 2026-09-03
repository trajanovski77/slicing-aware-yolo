"""Prepare the Kaggle "Ships in Aerial Images" dataset for Ultralytics YOLO detection.

The Kaggle export (siddharthkumarsah/ships-in-aerial-images) is already in YOLO
format: ``train/valid/test`` each with ``images/`` and ``labels/`` plus a
``data.yaml`` (single ``ship`` class). Preparation is therefore a thin adapter:
discover the export root, map ``valid`` -> ``val``, link images and labels into the
project's ``data/processed/ships`` layout, and emit ``configs/datasets/ships.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.datasets.validation import raise_on_invalid, validate_yolo_dataset
from src.utils.paths import IMAGE_EXTENSIONS, ensure_dir, link_or_copy, write_json, write_yaml

# Source split dir -> YOLO split name.
_SPLIT_MAP = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def yolo_label_line_to_hbb(line: str) -> str | None:
    """Normalise one YOLO label line to ``class cx cy w h`` (all in [0,1]).

    The Kaggle ships export mixes detection boxes (5 fields) with segmentation
    polygons (``class x1 y1 ... xn yn``). Coordinates are already normalised, so a
    polygon becomes its axis-aligned bounding box via the min/max of its points.
    """
    parts = line.split()
    if len(parts) < 5:
        return None
    cls = parts[0]
    coords = [float(v) for v in parts[1:]]
    if len(coords) == 4:  # already cx cy w h
        cx, cy, w, h = coords
    elif len(coords) >= 6 and len(coords) % 2 == 0:  # polygon: pairs of points
        xs = coords[0::2]
        ys = coords[1::2]
        x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
        cx, cy, w, h = (x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min
    else:
        return None
    if w <= 0 or h <= 0:
        return None
    clamp = lambda v: min(1.0, max(0.0, v))
    return f"{cls} " + " ".join(f"{clamp(v):.6f}" for v in (cx, cy, w, h))


def _convert_label_file(src: Path, dst: Path) -> None:
    lines = []
    if src.exists():
        for raw in src.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            converted = yolo_label_line_to_hbb(raw)
            if converted is not None:
                lines.append(converted)
    dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


@dataclass(frozen=True)
class ShipsPrepareSummary:
    output_dir: str
    train_images: int
    val_images: int
    test_images: int
    classes: list[str]


def _find_export_root(raw_dir: Path) -> Path:
    """Return the directory that directly contains the train/valid/test splits."""
    candidates = [raw_dir, *[p for p in sorted(raw_dir.glob("*")) if p.is_dir()]]
    for candidate in candidates:
        if any((candidate / split / "images").is_dir() for split in _SPLIT_MAP):
            return candidate
    raise FileNotFoundError(
        f"Could not find a train/valid/test export with images/ under {raw_dir}. "
        "Expected the Kaggle 'ships-in-aerial-images' layout."
    )


def _read_class_names(export_root: Path) -> list[str]:
    data_yaml = export_root / "data.yaml"
    if data_yaml.exists():
        with data_yaml.open("r", encoding="utf-8") as handle:
            meta = yaml.safe_load(handle) or {}
        names = meta.get("names")
        if isinstance(names, list) and names:
            return [str(n) for n in names]
        if isinstance(names, dict) and names:
            return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    return ["ship"]


def prepare_ships(
    raw_dir: str | Path,
    out_dir: str | Path,
    dataset_yaml: str | Path = "configs/datasets/ships.yaml",
    copy_images: bool = False,
    validate: bool = True,
) -> ShipsPrepareSummary:
    export_root = _find_export_root(Path(raw_dir))
    out_path = Path(out_dir)
    class_names = _read_class_names(export_root)

    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for src_split, yolo_split in _SPLIT_MAP.items():
        image_dir = export_root / src_split / "images"
        label_dir = export_root / src_split / "labels"
        if not image_dir.is_dir():
            continue
        ensure_dir(out_path / "images" / yolo_split)
        ensure_dir(out_path / "labels" / yolo_split)
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            link_or_copy(image_path, out_path / "images" / yolo_split / image_path.name, copy=copy_images)
            src_label = label_dir / f"{image_path.stem}.txt"
            dst_label = out_path / "labels" / yolo_split / f"{image_path.stem}.txt"
            # Always rewrite labels: the export mixes HBB and polygon lines; normalise to HBB.
            _convert_label_file(src_label, dst_label)
            counts[yolo_split] += 1

    yaml_data = {
        "path": str(out_path),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "task": "detect",
        "names": {idx: name for idx, name in enumerate(class_names)},
    }
    write_yaml(yaml_data, dataset_yaml)

    summary = ShipsPrepareSummary(
        output_dir=str(out_path),
        train_images=counts["train"],
        val_images=counts["val"],
        test_images=counts["test"],
        classes=class_names,
    )
    write_json(summary.__dict__, out_path / "metadata" / "prepare_summary.json")

    if validate:
        raise_on_invalid(validate_yolo_dataset(out_path, task="detect"))
    return summary
