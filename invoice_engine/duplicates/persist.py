"""Persist the detector's duplicate findings to SQLite.

Step 6 graded the scored pairs in memory but did not save them. Step 7 needs
both problem streams (validation + duplicates) in the database so the review
queue can merge them. This writes the duplicate pairs that clear the chosen
confidence threshold into a `duplicate_findings` table.

We store the per-signal scores alongside the blended confidence on purpose: they
ARE the human-readable reason ("matched on amount+date, invoice number differs"),
which the review queue surfaces to the AP clerk.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import polars as pl

from ..ingestion.store import DB_PATH

# Chosen in Step 6.3: top of the perfect-precision/recall band (0.70-0.90).
# Most conservative cut that still achieves recall 1.0 on the synthetic data.
CONFIDENCE_THRESHOLD = 0.90

_TABLE_SQL = """
DROP TABLE IF EXISTS duplicate_findings;
CREATE TABLE duplicate_findings (
    record_id_a INTEGER,
    record_id_b INTEGER,
    id_sim      REAL,
    name_sim    REAL,
    amount_sim  REAL,
    date_sim    REAL,
    confidence  REAL
);
"""


def store_duplicate_findings(
    scored: pl.DataFrame,
    db_path: Path = DB_PATH,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> pl.DataFrame:
    """Write pairs with confidence >= threshold to `duplicate_findings`.

    Returns the frame of stored findings (so callers can reuse it without a
    second DB round-trip).
    """
    findings = scored.filter(pl.col("confidence") >= threshold)
    con = sqlite3.connect(db_path)
    try:
        con.executescript(_TABLE_SQL)
        con.executemany(
            "INSERT INTO duplicate_findings "
            "(record_id_a, record_id_b, id_sim, name_sim, amount_sim, date_sim, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            findings.select(
                "record_id_a", "record_id_b",
                "id_sim", "name_sim", "amount_sim", "date_sim", "confidence",
            ).iter_rows(),
        )
        con.commit()
    finally:
        con.close()
    return findings
