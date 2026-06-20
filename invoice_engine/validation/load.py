"""Read the stored DB back into typed polars frames for validation.

Storage kept money + dates as TEXT (to stay exact). To *compute* on them we cast
back: money -> Decimal, date strings -> Date. This is the other half of the
"store exact text, parse when computing" decision from Step 4.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import polars as pl

from ..ingestion.store import DB_PATH


def _read_table(con: sqlite3.Connection, name: str) -> pl.DataFrame:
    cur = con.execute(f"SELECT * FROM {name}")
    cols = [d[0] for d in cur.description]
    return pl.DataFrame(cur.fetchall(), schema=cols, orient="row")


def load_for_validation(db_path: Path = DB_PATH) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return ``(invoices, line_items)`` with money/date columns properly typed."""
    con = sqlite3.connect(db_path)
    try:
        invoices = _read_table(con, "invoices")
        line_items = _read_table(con, "line_items")
    finally:
        con.close()

    invoices = invoices.with_columns(
        pl.col("invoice_amount").cast(pl.Decimal(scale=2)),
        pl.col("tax_amount").cast(pl.Decimal(scale=2)),
        pl.col("invoice_date").str.to_date(),
        pl.col("due_date").str.to_date(),
        pl.col("payment_date").str.to_date(),
    )
    line_items = line_items.with_columns(
        pl.col("unit_price").cast(pl.Decimal(scale=2)),
        pl.col("line_total").cast(pl.Decimal(scale=2)),
    )
    return invoices, line_items
