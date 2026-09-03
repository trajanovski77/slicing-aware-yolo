from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.datasets.converters import polygon_area
from src.utils.paths import ensure_dir, project_root, read_yaml


def size_bucket(area: float) -> str:
    if area < 0.01:
        return "small"
    if area < 0.09:
        return "medium"
    return "large"


def _dataset_root(data_yaml: str | Path) -> Path:
    config = read_yaml(data_yaml)
    root = Path(config.get("path", "."))
    if not root.is_absolute():
        root = project_root() / root
    return root


def _label_area(parts: list[str], task: str) -> float | None:
    if task == "detect" and len(parts) >= 5:
        return max(0.0, float(parts[3])) * max(0.0, float(parts[4]))
    if task == "obb" and len(parts) >= 9:
        points = [(float(parts[idx]), float(parts[idx + 1])) for idx in range(1, 9, 2)]
        return polygon_area(points)
    return None


def analyze_object_sizes(
    data_yaml: str | Path,
    split: str = "val",
    output_dir: str | Path = "results/tables",
) -> list[dict[str, Any]]:
    config = read_yaml(data_yaml)
    root = _dataset_root(data_yaml)
    task = str(config.get("task", "detect"))
    split_value = Path(config.get(split) or config.get("val") or f"images/{split}")
    if split_value.parts and split_value.parts[0] == "images":
        label_dir = root / "labels" / Path(*split_value.parts[1:])
    else:
        label_dir = root / "labels" / split
    counts: dict[tuple[int, str], int] = {}
    for label_path in sorted(label_dir.glob("*.txt")):
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            area = _label_area(parts, task)
            if area is None:
                continue
            key = (int(parts[0]), size_bucket(area))
            counts[key] = counts.get(key, 0) + 1
    names = config.get("names", {})
    rows = []
    for (class_id, bucket), count in sorted(counts.items()):
        class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
        rows.append(
            {
                "data": str(data_yaml),
                "split": split,
                "task": task,
                "class_id": class_id,
                "class_name": class_name,
                "size_bucket": bucket,
                "instances": count,
            }
        )
    output_path = ensure_dir(output_dir) / "object_size_distribution.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            handle.write("")
    return rows
