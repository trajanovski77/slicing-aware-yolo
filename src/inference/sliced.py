"""Slicing-aware (tiled) YOLO inference.

Geometry and merge are kept separate from the model glue so the coordinate round-trip
(slice -> per-tile detect -> offset -> merge) is unit-testable without torch/ultralytics.

- ``sliced_detections`` is the pure pipeline: it takes an image and a ``predict_local``
  callable (tile crops -> per-tile detections in *tile-local* coordinates), remaps every
  detection to original-image coordinates, and merges. Inject a fake ``predict_local`` in
  tests.
- ``UltralyticsTilePredictor`` is the real adapter that batches tile crops through a YOLO
  model. Built lazily so importing this module needs no GPU.

Reuses ``src.datasets.tiling.tile_windows`` (the same sliding-window generator the prior
repo used for training-time tiling) and ``src.inference.merge``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from src.datasets.tiling import TileWindow, tile_windows
from src.inference.merge import (
    Detection,
    detections_from_ultralytics,
    merge_detections,
)

# A local predictor maps a batch of tile crops (BGR arrays) to per-tile detections
# expressed in tile-local pixel coordinates.
PredictLocal = Callable[[Sequence[np.ndarray]], list[list[Detection]]]


@dataclass(frozen=True)
class SlicingPolicy:
    tile_size: int
    overlap: float = 0.0  # fraction of tile_size (e.g. 0.10, 0.20)
    merge_method: str = "nms_class_aware"
    merge_iou: float = 0.55
    include_full_image: bool = False

    @property
    def overlap_px(self) -> int:
        return round(self.overlap * self.tile_size)

    @property
    def label(self) -> str:
        return f"tile{self.tile_size}_ov{int(round(self.overlap * 100))}"


@dataclass(frozen=True)
class SlicedResult:
    detections: list[Detection]
    n_tiles: int
    raw_detection_count: int  # before merge (for duplicate-rate diagnostics, RQ5)

    @property
    def merged_detection_count(self) -> int:
        return len(self.detections)


def windows_for_image(width: int, height: int, policy: SlicingPolicy) -> list[TileWindow]:
    return tile_windows(width, height, policy.tile_size, policy.overlap_px)


def sliced_detections(
    image: np.ndarray,
    policy: SlicingPolicy,
    predict_local: PredictLocal,
    full_image_predict: Callable[[np.ndarray], list[Detection]] | None = None,
) -> SlicedResult:
    """Run the full sliced pipeline on one image and return merged, remapped detections."""
    height, width = image.shape[:2]
    windows = windows_for_image(width, height, policy)
    crops = [image[w.y : w.y2, w.x : w.x2] for w in windows]

    local_per_tile = predict_local(crops)
    if len(local_per_tile) != len(windows):
        raise ValueError(
            f"predict_local returned {len(local_per_tile)} results for {len(windows)} tiles"
        )

    remapped: list[Detection] = []
    for window, locals_ in zip(windows, local_per_tile):
        for det in locals_:
            x1, y1, x2, y2 = det.xyxy
            remapped.append(
                Detection(
                    class_id=det.class_id,
                    confidence=det.confidence,
                    xyxy=(x1 + window.x, y1 + window.y, x2 + window.x, y2 + window.y),
                    source=policy.label,
                    weight=det.weight,
                )
            )

    if policy.include_full_image and full_image_predict is not None:
        remapped.extend(full_image_predict(image))

    raw_count = len(remapped)
    merged = merge_detections(remapped, policy.merge_method, policy.merge_iou)
    return SlicedResult(detections=merged, n_tiles=len(windows), raw_detection_count=raw_count)


class UltralyticsTilePredictor:
    """Adapter turning a YOLO model into a batched ``PredictLocal`` over tile crops."""

    def __init__(
        self,
        model: Any,
        imgsz: int,
        conf: float = 0.25,
        iou: float = 0.7,
        batch_size: int = 8,
        device: Any = None,
        half: bool = False,
        max_det: int = 10000,
    ) -> None:
        self.model = model
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.batch_size = max(1, batch_size)
        self.device = device
        self.half = half
        # Per-TILE cap (Ultralytics default 300). A 1024 px tile of a dense DOTA scene can
        # hold more than 300 instances, so the cap is lifted here as well for symmetry.
        self.max_det = int(max_det)

    def _predict_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "half": self.half,
            "max_det": self.max_det,
            "verbose": False,
            "save": False,
        }
        if self.device not in (None, "auto", ""):
            kwargs["device"] = self.device
        return kwargs

    def __call__(self, crops: Sequence[np.ndarray]) -> list[list[Detection]]:
        results: list[list[Detection]] = []
        kwargs = self._predict_kwargs()
        for start in range(0, len(crops), self.batch_size):
            batch = list(crops[start : start + self.batch_size])
            outputs = self.model.predict(source=batch, **kwargs)
            for out in outputs:
                results.append(detections_from_ultralytics(out, offset=(0.0, 0.0)))
        return results
