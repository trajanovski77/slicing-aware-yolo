from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.utils.paths import ensure_dir, iter_images
from src.utils.ultralytics_env import configure_ultralytics_settings


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def _draw_predictions(model, image_path: Path, imgsz: int, conf: float) -> np.ndarray:
    result = model.predict(source=str(image_path), imgsz=imgsz, conf=conf, verbose=False)[0]
    return result.plot()


def create_disagreement_panels(
    weights: dict[str, str],
    image_dir: str | Path,
    output_dir: str | Path = "results/predictions/error_analysis",
    task: str = "detect",
    imgsz: int = 640,
    sample_count: int = 24,
    conf: float = 0.25,
) -> list[Path]:
    configure_ultralytics_settings()
    from ultralytics import YOLO

    images = iter_images(image_dir)[:sample_count]
    if not images:
        raise FileNotFoundError(f"No images found for error analysis: {image_dir}")
    models = {name: YOLO(path, task=task) for name, path in weights.items()}
    out_root = ensure_dir(output_dir)
    outputs: list[Path] = []

    for image_path in images:
        panels = []
        labels = []
        for name, model in models.items():
            panel = _draw_predictions(model, image_path, imgsz, conf)
            panels.append(panel)
            labels.append(name)
        height = max(panel.shape[0] for panel in panels)
        normalized = []
        for label, panel in zip(labels, panels):
            scale = height / panel.shape[0]
            resized = cv2.resize(panel, (int(panel.shape[1] * scale), height))
            cv2.putText(
                resized,
                label,
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                resized,
                label,
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            normalized.append(resized)
        canvas = cv2.hconcat(normalized)
        output_path = out_root / f"{image_path.stem}_disagreement.jpg"
        cv2.imwrite(str(output_path), canvas)
        outputs.append(output_path)
    return outputs
