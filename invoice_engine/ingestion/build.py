"""Build the SQLite database from the raw CSVs.

Run it:  python -m invoice_engine.ingestion.build

Pipeline: read_raw (typed load) -> clean_frames (add normalized name) ->
write_sqlite. Then prints checks that prove the important things survived.
"""

from __future__ import annotations

import sqlite3
import sys

import polars as pl

from ..cleaning.normalize import clean_frames
from .load import read_raw
from .store import DB_PATH, write_sqlite


def build_database():
    frames = read_raw()
    frames = clean_frames(frames)
    db_path = write_sqlite(frames)
    return db_path, frames


def _verify(db_path, frames):
    con = sqlite3.connect(db_path)
    try:
        print("=== ROW COUNTS (SQLite) ===")
        for table in ["parties", "invoices", "line_items", "ground_truth"]:
            n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table:13}: {n}")

        print("\n=== LEADING ZERO SURVIVED? (read back from SQLite) ===")
        rows = con.execute(
            "SELECT record_id, invoice_id, invoice_amount FROM invoices "
            "WHERE invoice_id LIKE '0%' LIMIT 3"
        ).fetchall()
        for r in rows:
            print(f"  record {r[0]}: invoice_id={r[1]!r}  amount={r[2]!r}")
        print("  (invoice_id is text + amount is exact text - both preserved)")
    finally:
        con.close()

    print("\n=== NORMALIZED NAME: original -> normalized ===")
    sample = (
        frames["invoices"]
        .select("party_name", "party_name_norm")
        .unique()
        .head(5)
    )
    with pl.Config(tbl_width_chars=200):
        print(sample)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    db_path, frames = build_database()
    _verify(db_path, frames)
    print(f"\nDatabase written to: {db_path.resolve()}")
