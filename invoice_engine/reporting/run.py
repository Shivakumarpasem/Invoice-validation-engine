"""Entry point for the CLI report.

Run it:
  python -m invoice_engine.reporting.run            # the sectioned briefing
  python -m invoice_engine.reporting.run --all      # + full worklist in terminal
  python -m invoice_engine.reporting.run --csv out.csv   # + export full worklist
  python -m invoice_engine.reporting.run --top 20   # show top 20 cases

Reads the already-built findings/queue from SQLite and renders them. Assumes the
pipeline has run: generator.generate -> ingestion.build -> validation.run ->
scoring.run (which persists duplicate_findings).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..ingestion.store import DB_PATH
from .report import render_report


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Invoice Validation Engine - review report")
    p.add_argument("--all", action="store_true", help="dump the full worklist to the terminal")
    p.add_argument("--csv", type=Path, default=None, help="write the full worklist to a CSV file")
    p.add_argument("--top", type=int, default=10, help="how many top cases to show (default 10)")
    return p.parse_args()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    args = _parse_args()
    render_report(DB_PATH, top_n=args.top, show_all=args.all, csv_path=args.csv)
