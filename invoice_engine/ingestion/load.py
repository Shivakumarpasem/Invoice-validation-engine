"""Load the raw CSVs into typed polars frames.

The single most important job here: force column types on read. A CSV has no
types, so polars would *guess* - and it would read invoice_id "0412809" as the
integer 412809, silently destroying the leading-zero forgery we planted. We
override the types instead of guessing.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

RAW_DIR = Path("data/raw")

# Columns we must NOT let polars auto-infer.
_INVOICE_OVERRIDES = {
    "record_id": pl.Int64,
    "invoice_id": pl.String,   # keep leading zeros - the whole point
    "party_id": pl.String,
    "po_number": pl.String,
    "invoice_amount": pl.Decimal(scale=2),  # exact money, not float
    "tax_amount": pl.Decimal(scale=2),
    "gl_account_code": pl.String,           # "5000" is a code, not a number
    "cost_center": pl.String,
}
_LINE_OVERRIDES = {
    "record_id": pl.Int64,
    "unit_price": pl.Decimal(scale=2),
    "line_total": pl.Decimal(scale=2),
}


def read_raw(raw_dir: Path = RAW_DIR) -> dict[str, pl.DataFrame]:
    """Read the four CSVs into a dict of frames keyed by table name."""
    invoices = pl.read_csv(
        raw_dir / "invoices.csv",
        schema_overrides=_INVOICE_OVERRIDES,
        try_parse_dates=True,  # invoice_date / due_date / payment_date -> Date
    )
    line_items = pl.read_csv(raw_dir / "line_items.csv", schema_overrides=_LINE_OVERRIDES)
    ground_truth = pl.read_csv(raw_dir / "ground_truth.csv")
    parties = pl.read_csv(
        raw_dir / "parties.csv",
        schema_overrides={"party_id": pl.String, "gl_account_code": pl.String},
    )
    return {
        "parties": parties,
        "invoices": invoices,
        "line_items": line_items,
        "ground_truth": ground_truth,
    }
