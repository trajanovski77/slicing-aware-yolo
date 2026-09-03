from __future__ import annotations

import numpy as np
import pytest

from src.eval.coco_eval import pr_f1, run_coco_eval


def _tiny_gt():
    return {
        "images": [
            {"id": 0, "file_name": "a", "width": 100, "height": 100},
            {"id": 1, "file_name": "b", "width": 100, "height": 100},
        ],
        "annotations": [
            {"id": 1, "image_id": 0, "category_id": 0, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
            {"id": 2, "image_id": 0, "category_id": 1, "bbox": [50, 50, 10, 10], "area": 100, "iscrowd": 0},
            {"id": 3, "image_id": 1, "category_id": 0, "bbox": [0, 0, 30, 30], "area": 900, "iscrowd": 0},
        ],
        "categories": [{"id": 0, "name": "plane"}, {"id": 1, "name": "ship"}],
    }


def _perfect_dt():
    return [
        {"image_id": 0, "category_id": 0, "bbox": [10, 10, 20, 20], "score": 0.99},
        {"image_id": 0, "category_id": 1, "bbox": [50, 50, 10, 10], "score": 0.95},
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 30, 30], "score": 0.98},
    ]


def test_pr_f1_perfect():
    prf = pr_f1(_tiny_gt(), _perfect_dt(), iou_threshold=0.5)
    assert prf["precision"] == pytest.approx(1.0)
    assert prf["recall"] == pytest.approx(1.0)
    assert prf["f1"] == pytest.approx(1.0)
    assert prf["tp"] == 3 and prf["fp"] == 0 and prf["n_gt"] == 3


def test_pr_f1_with_false_positive():
    dt = _perfect_dt() + [{"image_id": 1, "category_id": 0, "bbox": [80, 80, 5, 5], "score": 0.9}]
    prf = pr_f1(_tiny_gt(), dt, iou_threshold=0.5)
    assert prf["fp"] == 1
    assert prf["recall"] == pytest.approx(1.0)
    assert prf["precision"] == pytest.approx(3 / 4)


def test_run_coco_eval_perfect():
    pytest.importorskip("pycocotools")
    out = run_coco_eval(_tiny_gt(), _perfect_dt())
    assert out["map50"] > 0.99
    assert out["map50_95"] > 0.99
    assert {p["class_name"] for p in out["per_class"]} == {"plane", "ship"}


def test_run_coco_eval_empty_detections():
    pytest.importorskip("pycocotools")
    out = run_coco_eval(_tiny_gt(), [])
    assert out["map50"] == 0.0
    assert out["map50_95"] == 0.0


def test_coco_gt_from_yolo(tmp_path):
    import cv2

    from src.eval.coco_gt import build_from_yolo

    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    cv2.imwrite(str(images / "img1.png"), np.zeros((200, 100, 3), dtype=np.uint8))
    # class 0, centered box occupying the middle 50% in x and y.
    (labels / "img1.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    yaml_path = tmp_path / "ds.yaml"
    yaml_path.write_text("path: .\nval: images\nnames:\n  0: ship\n", encoding="utf-8")
    coco = build_from_yolo(images, labels, yaml_path)

    assert len(coco["images"]) == 1
    assert coco["images"][0]["width"] == 100 and coco["images"][0]["height"] == 200
    assert len(coco["annotations"]) == 1
    x, y, w, h = coco["annotations"][0]["bbox"]
    assert (x, y, w, h) == pytest.approx((25.0, 50.0, 50.0, 100.0))
