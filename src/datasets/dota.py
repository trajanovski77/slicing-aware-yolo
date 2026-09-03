from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.datasets.converters import (
    DOTA_CLASSES,
    DOTA_V15_CLASSES,
    HBBAnnotation,
    hbb_to_yolo_line,
    obb_to_yolo_line,
    parse_dota_annotation_line,
)
from src.datasets.tiling import annotations_for_tile, save_tile, tile_windows
from src.datasets.validation import raise_on_invalid, validate_yolo_dataset
from src.utils.paths import ensure_dir, iter_images, write_json, write_yaml


@dataclass(frozen=True)
class DotaPrepareSummary:
    output_dir: str
    task: str
    train_tiles: int
    val_tiles: int
    test_tiles: int
    tile_size: int
    overlap: int
    classes: list[str]


def _find_split_dirs(
    raw_dir: Path, split: str, task: str = "detect"
) -> tuple[Path | None, Path | None]:
    image_candidates = [
        raw_dir / split / "images",
        raw_dir / split / "Images",
        raw_dir / "images" / split,
        raw_dir / f"{split}_images",
        raw_dir / split,
    ]
    hbb_label_candidates = [
        raw_dir / split / "labelTxt-v1.5" / f"DOTA-v1.5_{split}_hbb",
        raw_dir / split / "labelTxt-v1.0" / f"DOTA-v1.0_{split}_hbb",
        raw_dir / split / "labelTxt" / f"DOTA-v1.5_{split}_hbb",
        raw_dir / split / "labelTxt" / f"DOTA-v1.0_{split}_hbb",
        raw_dir / split / "labelTxt_hbb",
        raw_dir / split / "labels_hbb",
        raw_dir / "labels_hbb" / split,
        raw_dir / f"{split}_labelTxt_hbb",
    ]
    obb_label_candidates = [
        raw_dir / split / "labelTxt",
        raw_dir / split / "labelTxt-v1.5" / f"DOTA-v1.5_{split}",
        raw_dir / split / "labelTxt-v1.0" / f"DOTA-v1.0_{split}",
        raw_dir / split / "labelTxt-v1.0",
        raw_dir / split / "labelTxt-v1.5",
        raw_dir / split / "labels",
        raw_dir / "labels" / split,
        raw_dir / "labelTxt" / split,
        raw_dir / f"{split}_labelTxt",
    ]
    label_candidates = (
        hbb_label_candidates + obb_label_candidates
        if task == "detect"
        else obb_label_candidates + hbb_label_candidates
    )
    image_dir = next((p for p in image_candidates if p.exists() and iter_images(p)), None)
    label_dir = next((p for p in label_candidates if p.exists()), None)
    return image_dir, label_dir


def _discover_dota_classes(label_dirs: list[Path]) -> list[str]:
    for label_dir in label_dirs:
        if not label_dir:
            continue
        for path in label_dir.glob("*.txt"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "container-crane" in text or "container crane" in text:
                return DOTA_V15_CLASSES
    return DOTA_CLASSES


def parse_dota_label(path: str | Path, classes: list[str] | None = None) -> list:
    annotations = []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        parsed = parse_dota_annotation_line(line, classes or DOTA_CLASSES)
        if parsed is not None:
            annotations.append(parsed)
    return annotations


def _obb_annotation_to_hbb_line(annotation, width: int, height: int) -> str | None:
    xs = [point[0] for point in annotation.points]
    ys = [point[1] for point in annotation.points]
    try:
        hbb = HBBAnnotation(
            class_name=annotation.class_name,
            class_id=annotation.class_id,
            x_min=min(xs),
            y_min=min(ys),
            x_max=max(xs),
            y_max=max(ys),
        )
        return hbb_to_yolo_line(hbb, width, height)
    except ValueError:
        return None


def _format_tile_labels(annotations: list, width: int, height: int, task: str) -> list[str]:
    if task == "obb":
        return [obb_to_yolo_line(item, width, height) for item in annotations]
    lines = []
    for item in annotations:
        line = _obb_annotation_to_hbb_line(item, width, height)
        if line is not None:
            lines.append(line)
    return lines


def prepare_dota(
    raw_dir: str | Path,
    out_dir: str | Path,
    dataset_yaml: str | Path = "configs/datasets/dota.yaml",
    task: str = "detect",
    tile_size: int = 1024,
    overlap: int = 200,
    min_area_ratio: float = 0.1,
    labels_only: bool = False,
    validate: bool = True,
) -> DotaPrepareSummary:
    if task not in {"detect", "obb"}:
        raise ValueError("task must be 'detect' or 'obb'")
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"DOTA root not found: {root}")
    out_path = Path(out_dir)
    metadata_path = out_path / "metadata" / "tiles.jsonl"
    if metadata_path.exists():
        metadata_path.unlink()

    counts: dict[str, int] = {}
    metadata_rows: list[str] = []
    split_dirs = {
        split: _find_split_dirs(root, split, task=task) for split in ("train", "val", "test")
    }
    classes = _discover_dota_classes(
        [
            label_dir
            for split, (_, label_dir) in split_dirs.items()
            if split != "test" and label_dir is not None
        ]
    )
    for split in ("train", "val", "test"):
        image_dir, label_dir = split_dirs[split]
        counts[split] = 0
        if image_dir is None:
            continue
        ensure_dir(out_path / "images" / split)
        ensure_dir(out_path / "labels" / split)
        for image_path in iter_images(image_dir):
            label_path = label_dir / f"{image_path.stem}.txt" if label_dir else None
            if split != "test" and (label_path is None or not label_path.exists()):
                raise FileNotFoundError(f"Missing DOTA label for {image_path.name}: {label_path}")
            annotations = (
                parse_dota_label(label_path, classes)
                if label_path and label_path.exists()
                else []
            )

            import cv2

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            height, width = image.shape[:2]
            for idx, window in enumerate(tile_windows(width, height, tile_size, overlap)):
                tile_name = (
                    f"{image_path.stem}__x{window.x}_y{window.y}_w{window.width}_"
                    f"h{window.height}{image_path.suffix}"
                )
                tile_image = out_path / "images" / split / tile_name
                tile_label = out_path / "labels" / split / f"{Path(tile_name).stem}.txt"
                if labels_only:
                    if not tile_image.exists():
                        raise FileNotFoundError(
                            f"Missing existing tile image for labels-only mode: {tile_image}"
                        )
                else:
                    save_tile(image_path, tile_image, window)
                tile_annotations = annotations_for_tile(annotations, window, min_area_ratio)
                label_lines = _format_tile_labels(
                    tile_annotations, window.width, window.height, task
                )
                tile_label.write_text(
                    "\n".join(label_lines) + ("\n" if label_lines else ""),
                    encoding="utf-8",
                )
                metadata_rows.append(
                    {
                        "tile_id": Path(tile_name).stem,
                        "split": split,
                        "source_image": image_path.name,
                        "source_stem": image_path.stem,
                        "tile_index": idx,
                        "x": window.x,
                        "y": window.y,
                        "width": window.width,
                        "height": window.height,
                        "task": task,
                    }
                )
                counts[split] += 1

    if counts.get("train", 0) == 0 or counts.get("val", 0) == 0:
        raise FileNotFoundError(
            f"Could not find DOTA train/val images under {root}. "
            "Expected train/images and val/images."
        )

    ensure_dir(metadata_path.parent)
    with metadata_path.open("w", encoding="utf-8") as handle:
        import json

        for row in metadata_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    yaml_data = {
        "path": out_path.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "task": task,
        "names": {idx: name for idx, name in enumerate(classes)},
    }
    write_yaml(yaml_data, dataset_yaml)
    summary = DotaPrepareSummary(
        output_dir=str(out_path),
        task=task,
        train_tiles=counts.get("train", 0),
        val_tiles=counts.get("val", 0),
        test_tiles=counts.get("test", 0),
        tile_size=tile_size,
        overlap=overlap,
        classes=classes,
    )
    write_json(summary.__dict__, out_path / "metadata" / "prepare_summary.json")
    if validate:
        raise_on_invalid(validate_yolo_dataset(out_path, task=task))
    return summary
