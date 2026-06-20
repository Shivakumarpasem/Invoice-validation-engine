"""Top-level entry point: build the whole synthetic dataset and save it.

Run it:  python -m invoice_engine.generator.generate

Pipeline: config -> party pool -> clean invoices -> plant messiness -> write CSVs.
Output lands in data/raw/ (gitignored). These CSVs are the 'raw feed' that the
Step 4 ingestion layer will load, clean, and store in SQLite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

from .config import GeneratorConfig
from .core import generate_clean_invoices
from .messiness import inject_messiness
from .pool import build_party_pool

RAW_DIR = Path("data/raw")


def generate_dataset(config: GeneratorConfig | None = None, write: bool = True):
    """Build (and optionally write) the full dataset.

    Returns ``(parties, invoices, line_items, ground_truth)`` as polars frames.
    """
    config = config or GeneratorConfig()

    parties = build_party_pool(config)
    invoices, line_items = generate_clean_invoices(config, parties)
    invoices, line_items, ground_truth = inject_messiness(invoices, line_items, config)

    if write:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        parties.write_csv(RAW_DIR / "parties.csv")
        invoices.write_csv(RAW_DIR / "invoices.csv")
        line_items.write_csv(RAW_DIR / "line_items.csv")
        ground_truth.write_csv(RAW_DIR / "ground_truth.csv")

    return parties, invoices, line_items, ground_truth


def _print_summary(parties, invoices, line_items, ground_truth):
    print("=== DATASET BUILT ===")
    print(f"parties      : {parties.height}")
    print(f"invoices     : {invoices.height}  (clean + planted copies)")
    print(f"line items   : {line_items.height}")
    print(f"problems     : {ground_truth.height}  "
          f"({100 * ground_truth.height / invoices.height:.1f}% of invoices)")

    print("\n=== PROBLEM BREAKDOWN (the answer key) ===")
    breakdown = (
        ground_truth.group_by("problem_family", "problem_type")
        .len()
        .sort("problem_family", "problem_type")
    )
    print(breakdown)

    # Show one leading-zero forgery next to the original it copies.
    lz = ground_truth.filter(pl.col("problem_type") == "leading_zero").head(1)
    if lz.height:
        fake_id = lz["record_id"][0]
        orig_id = lz["duplicate_of"][0]
        pair = invoices.filter(pl.col("record_id").is_in([orig_id, fake_id]))
        print("\n=== EXAMPLE: a leading-zero forgery vs its original ===")
        with pl.Config(tbl_cols=-1, tbl_width_chars=200):
            print(pair.select(
                "record_id", "invoice_id", "party_name", "invoice_date", "invoice_amount"
            ))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    data = generate_dataset()
    _print_summary(*data)
    print(f"\nWritten to: {RAW_DIR.resolve()}")
