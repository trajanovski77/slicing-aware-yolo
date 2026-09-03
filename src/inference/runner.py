"""Shared parsing of a slicing config into concrete inference variants.

Keeps predict / benchmark / diagnostics scripts in lock-step: they all enumerate the same
native-imgsz points and sliced policies from one config.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from src.inference.sliced import SlicingPolicy
from src.utils.paths import iter_images, read_yaml


@dataclass(frozen=True)
class Variant:
    kind: str            # "native" or "sliced"
    label: str           # e.g. "native1024" or "tile512_ov10"
    imgsz: int | None = None            # native only
    policy: SlicingPolicy | None = None  # sliced only


def load_slicing_config(path: str | Path) -> dict[str, Any]:
    return read_yaml(path)


def build_policy(policy_cfg: dict[str, Any], merge_cfg: dict[str, Any]) -> SlicingPolicy:
    return SlicingPolicy(
        tile_size=int(policy_cfg["tile"]),
        overlap=float(policy_cfg.get("overlap", 0.0)),
        merge_method=str(merge_cfg.get("method", "nms_class_aware")),
        merge_iou=float(merge_cfg.get("iou", 0.55)),
        include_full_image=bool(policy_cfg.get("include_full_image", False)),
    )


def iter_variants(config: dict[str, Any]) -> Iterator[Variant]:
    for imgsz in config.get("native_imgsz", []):
        yield Variant(kind="native", label=f"native{int(imgsz)}", imgsz=int(imgsz))
    merge_cfg = config.get("merge", {})
    for policy_cfg in config.get("policies", []):
        policy = build_policy(policy_cfg, merge_cfg)
        yield Variant(kind="sliced", label=policy.label, policy=policy)


def resolve_images(config: dict[str, Any]) -> list[Path]:
    """Locate the original-resolution val images the study runs on."""
    fmt = str(config.get("gt_format", "dota_raw"))
    split = str(config.get("split", "val"))
    if fmt == "dota_raw":
        from src.datasets.dota import _find_split_dirs

        image_dir, _ = _find_split_dirs(Path(config["raw_dir"]), split, task="detect")
        if image_dir is None:
            raise FileNotFoundError(f"No DOTA {split} images under {config['raw_dir']}")
        return iter_images(image_dir)
    if fmt == "yolo":
        meta = read_yaml(config["dataset"])
        root = Path(config.get("images_root", meta.get("path", ".")))
        images_dir = config.get("images_dir") or (root / meta.get(split, f"images/{split}"))
        return iter_images(images_dir)
    raise ValueError(f"Unknown gt_format '{fmt}'")
