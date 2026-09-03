"""Native full-image YOLO inference — the deployment baseline.

Ultralytics resizes the (possibly huge) original image to a single ``imgsz`` and runs one
forward pass. This is the standard setting the slicing study compares against; running it
at several ``imgsz`` values isolates the effect of slicing from the effect of raw
inference resolution.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.inference.merge import Detection, detections_from_ultralytics


def native_detections(
    model: Any,
    image: np.ndarray,
    imgsz: int,
    conf: float = 0.25,
    iou: float = 0.7,
    device: Any = None,
    half: bool = False,
    max_det: int = 10000,
) -> list[Detection]:
    """Single full-image forward pass -> detections in original-image coordinates.

    ``max_det`` overrides Ultralytics' default cap of 300 boxes per image, which silently
    truncates the native output on dense DOTA scenes (48/458 val scenes exceed 300 GT).
    """
    kwargs: dict[str, Any] = {
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "half": half,
        "max_det": int(max_det),
        "verbose": False,
        "save": False,
    }
    if device not in (None, "auto", ""):
        kwargs["device"] = device
    result = model.predict(source=image, **kwargs)[0]
    return detections_from_ultralytics(result, offset=(0.0, 0.0), source=f"native{imgsz}")
