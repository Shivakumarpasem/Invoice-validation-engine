"""Build and show the unified review queue.

Run it:  python -m invoice_engine.scoring.run

Pipeline: run the duplicate detector -> persist the duplicates that clear the
threshold -> merge with validation findings into one ranked row-per-invoice
queue -> print the top of the worklist and the clean/needs-review split.

Assumes validation has already run (validation_findings exists). Rebuild order:
  generator.generate -> ingestion.build -> validation.run -> scoring.run
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

from ..duplicates.persist import CONFIDENCE_THRESHOLD, store_duplicate_findings
from ..duplicates.run import detect
from ..ingestion.store import DB_PATH
from .queue import build_queue, queue_summary


def run(db_path: Path = DB_PATH) -> pl.DataFrame:
    scored = detect(db_path)
    store_duplicate_findings(scored, db_path, CONFIDENCE_THRESHOLD)
    return build_queue(db_path)


def _report(db_path: Path = DB_PATH) -> None:
    queue = run(db_path)
    summary = queue_summary(queue, db_path)

    print("=== REVIEW QUEUE ===")
    print(f"total invoices : {summary['total_invoices']}")
    print(f"needs review   : {summary['needs_review']}")
    print(f"clean          : {summary['clean']}  "
          f"({100 * summary['clean'] / summary['total_invoices']:.1f}% auto-approvable)\n")

    print(f"duplicate threshold applied: {CONFIDENCE_THRESHOLD}\n")

    print("=== TOP OF THE QUEUE (worst first) ===")
    with pl.Config(fmt_str_lengths=90, tbl_width_chars=220, tbl_rows=12):
        print(queue.select(
            "record_id", "invoice_id", "party_name", "priority", "n_problems", "reasons"
        ).head(12))

    print("\n=== PRIORITY BANDS ===")
    bands = queue.with_columns(
        pl.when(pl.col("priority") >= 0.90).then(pl.lit("duplicate (>=0.90)"))
        .when(pl.col("priority") >= 0.80).then(pl.lit("validation error (0.80)"))
        .otherwise(pl.lit("warning (<0.80)"))
        .alias("band")
    )
    print(bands.group_by("band").len().sort("band"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    _report()
