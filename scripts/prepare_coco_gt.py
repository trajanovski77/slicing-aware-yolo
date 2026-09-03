#!/usr/bin/env python3
"""Build and cache a COCO ground-truth JSON for a slicing-study dataset config.

    python scripts/prepare_coco_gt.py --config configs/slicing/dota.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.coco_gt import build_coco_gt, write_coco_gt  # noqa: E402
from src.inference.runner import load_slicing_config  # noqa: E402
from src.utils.logging import configure_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build COCO GT from a slicing config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None, help="Defaults to results/coco/<name>_<split>_gt.json")
    args = parser.parse_args()
    logger = configure_logging()

    config = load_slicing_config(args.config)
    name = Path(args.config).stem
    split = str(config.get("split", "val"))
    output = args.output or (ROOT / "results" / "coco" / f"{name}_{split}_gt.json")

    coco = build_coco_gt(config)
    write_coco_gt(coco, output)
    logger.info(
        "Wrote COCO GT: %d images, %d annotations, %d categories -> %s",
        len(coco["images"]), len(coco["annotations"]), len(coco["categories"]), output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
