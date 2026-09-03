#!/usr/bin/env python3
"""Emit Table 1 (native vs sliced) per dataset as CSV + LaTeX.

    python scripts/emit_slicing_tables.py --datasets "dota ships"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reporting.slicing_tables import write_table1  # noqa: E402
from src.utils.logging import configure_logging  # noqa: E402
from src.utils.paths import ensure_dir  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit slicing Table 1 per dataset.")
    parser.add_argument("--datasets", default="dota ships")
    parser.add_argument("--tables-dir", default=str(ROOT / "results" / "tables"))
    parser.add_argument("--speed-dir", default=str(ROOT / "results" / "metrics" / "speed"))
    parser.add_argument("--paper-dir", default=str(ROOT / "paper_assets" / "tables"))
    args = parser.parse_args()
    logger = configure_logging()

    tables_dir = Path(args.tables_dir)
    paper_dir = ensure_dir(Path(args.paper_dir))
    for ds in args.datasets.split():
        metrics_csv = tables_dir / f"{ds}_slicing_metrics.csv"
        speed_csv = Path(args.speed_dir) / ds / f"{ds}_speed.csv"
        if not metrics_csv.exists():
            logger.warning("No metrics for %s (%s); skipping", ds, metrics_csv)
            continue
        write_table1(
            metrics_csv, speed_csv,
            out_csv=tables_dir / f"table1_{ds}.csv",
            out_tex=paper_dir / f"table1_{ds}.tex",
            caption=f"Native vs. slicing-aware inference on {ds.upper()} (per model and slicing policy).",
            label=f"tab:table1_{ds}")
        logger.info("Table 1 for %s -> %s", ds, paper_dir / f"table1_{ds}.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
