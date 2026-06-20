"""Write the cleaned frames into a SQLite database.

Uses Python's built-in ``sqlite3`` (no extra dependency, fully self-contained).

Money + date storage choice: SQLite has no exact-decimal or date type. Its
NUMERIC affinity would turn our Decimal into a float (REAL) - reintroducing the
rounding error we deliberately avoided. So money and dates are stored as TEXT
(exact decimal strings / ISO date strings) and parsed back to Decimal/Date when
we compute. Keys and quantities stay INTEGER.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import polars as pl

DB_PATH = Path("data/invoices.db")

# line_id is the only column SQLite generates itself (AUTOINCREMENT).
_SCHEMA_SQL = """
CREATE TABLE parties (
    party_id              TEXT PRIMARY KEY,
    party_name            TEXT NOT NULL,
    direction             TEXT NOT NULL,
    default_payment_terms TEXT,
    gl_account_code       TEXT,
    cost_center           TEXT,
    party_name_norm       TEXT
);
CREATE TABLE invoices (
    record_id        INTEGER PRIMARY KEY,
    invoice_id       TEXT NOT NULL,      -- printed number, may repeat
    direction        TEXT NOT NULL,
    party_id         TEXT,
    party_name       TEXT,
    party_name_norm  TEXT,
    invoice_date     TEXT,
    due_date         TEXT,
    payment_date     TEXT,
    invoice_amount   TEXT,               -- exact decimal as text
    tax_amount       TEXT,
    currency         TEXT,
    payment_terms    TEXT,
    po_number        TEXT,
    status           TEXT,
    gl_account_code  TEXT,
    cost_center      TEXT
);
CREATE TABLE line_items (
    line_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id   INTEGER NOT NULL,        -- FK -> invoices.record_id
    description TEXT,
    qty         INTEGER,
    unit_price  TEXT,
    line_total  TEXT
);
CREATE TABLE ground_truth (
    record_id      INTEGER,
    problem_family TEXT,
    problem_type   TEXT,
    duplicate_of   INTEGER,
    detail         TEXT
);
"""

_TABLE_ORDER = ["parties", "invoices", "line_items", "ground_truth"]


def _to_text(df: pl.DataFrame) -> pl.DataFrame:
    """Cast Decimal and Date columns to String so SQLite stores them exactly."""
    casts = [
        pl.col(name).cast(pl.String)
        for name, dtype in df.schema.items()
        if dtype == pl.Date or isinstance(dtype, pl.Decimal)
    ]
    return df.with_columns(casts) if casts else df


def write_sqlite(frames: dict[str, pl.DataFrame], db_path: Path = DB_PATH) -> Path:
    """Create the DB fresh and insert all four tables. Returns the db path."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()  # rebuild from scratch each run (idempotent)

    con = sqlite3.connect(db_path)
    try:
        con.executescript(_SCHEMA_SQL)
        for table in _TABLE_ORDER:
            df = _to_text(frames[table])
            cols = df.columns
            placeholders = ", ".join("?" * len(cols))
            con.executemany(
                f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})",
                df.iter_rows(),
            )
        con.commit()
    finally:
        con.close()
    return db_path
