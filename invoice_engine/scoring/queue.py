"""Build the unified review queue - one ranked row per invoice.

This is the project's headline output: instead of two separate problem logs
(validation findings per-invoice, duplicate findings per-pair), we collapse
everything to ONE row per flagged invoice - the way a real AP clerk works a
worklist ("which invoice do I look at next?").

Two reconciliations happen here:
  1. Shape: a duplicate is a PAIR, so it becomes a reason attached to BOTH
     invoices in the pair (each invoice's row says "duplicate of record X").
  2. Ranking: validation has a 0/1 severity, duplicates have a 0-1 confidence.
     We map both onto one `priority` in [0,1] so the queue has a single sort key.

Priority rule (deliberately simple and defensible; see DECISIONS.md):
  * duplicate        -> priority = its confidence (0.90-1.0). The headline risk.
  * validation error -> 0.80 (clearly wrong: broken sum, impossible date).
  * validation warn  -> 0.30 (needs a human: missing PO).
An invoice with several problems takes the MAX (its worst issue) but lists ALL
reasons, so nothing is hidden.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import polars as pl

from ..ingestion.store import DB_PATH

_PRIORITY_ERROR = 0.80
_PRIORITY_WARNING = 0.30


def _read(con: sqlite3.Connection, name: str) -> pl.DataFrame:
    cur = con.execute(f"SELECT * FROM {name}")
    cols = [d[0] for d in cur.description]
    return pl.DataFrame(cur.fetchall(), schema=cols, orient="row")


def _validation_reasons(findings: pl.DataFrame) -> pl.DataFrame:
    """validation_findings -> per-invoice (record_id, reason, priority)."""
    if findings.is_empty():
        return pl.DataFrame(schema={"record_id": pl.Int64, "reason": pl.String, "priority": pl.Float64})
    return findings.select(
        pl.col("record_id"),
        pl.format("{}: {}", "check_name", "detail").alias("reason"),
        pl.when(pl.col("severity") == "error")
        .then(_PRIORITY_ERROR)
        .otherwise(_PRIORITY_WARNING)
        .alias("priority"),
    )


def _duplicate_reasons(findings: pl.DataFrame) -> pl.DataFrame:
    """duplicate_findings (a pair) -> TWO per-invoice rows (one for each side)."""
    if findings.is_empty():
        return pl.DataFrame(schema={"record_id": pl.Int64, "reason": pl.String, "priority": pl.Float64})

    # Side A: this invoice is a likely duplicate of B; symmetric for side B.
    side_a = findings.select(
        pl.col("record_id_a").alias("record_id"),
        pl.format(
            "duplicate: likely same bill as record {} (confidence {})",
            "record_id_b", pl.col("confidence").round(2),
        ).alias("reason"),
        pl.col("confidence").alias("priority"),
    )
    side_b = findings.select(
        pl.col("record_id_b").alias("record_id"),
        pl.format(
            "duplicate: likely same bill as record {} (confidence {})",
            "record_id_a", pl.col("confidence").round(2),
        ).alias("reason"),
        pl.col("confidence").alias("priority"),
    )
    return pl.concat([side_a, side_b], how="vertical")


def build_queue(db_path: Path = DB_PATH) -> pl.DataFrame:
    """Collapse both finding streams into one ranked row per flagged invoice."""
    con = sqlite3.connect(db_path)
    try:
        invoices = _read(con, "invoices")
        validation = _read(con, "validation_findings")
        duplicates = _read(con, "duplicate_findings")
    finally:
        con.close()

    reasons = pl.concat(
        [_validation_reasons(validation), _duplicate_reasons(duplicates)],
        how="vertical",
    )

    # One row per invoice: gather all its reasons, take its worst priority.
    per_invoice = reasons.group_by("record_id").agg(
        pl.col("reason"),                       # list of reason strings
        pl.col("priority").max().alias("priority"),
        pl.len().alias("n_problems"),
    )

    # Attach a little invoice context for the report, then rank worst-first.
    queue = (
        per_invoice.join(
            invoices.select("record_id", "invoice_id", "party_name", "invoice_amount"),
            on="record_id",
            how="left",
        )
        .with_columns(
            pl.col("reason").list.join(" | ").alias("reasons"),
            pl.lit("needs_review").alias("verdict"),
        )
        .drop("reason")
        .sort("priority", descending=True)
    )
    return queue


def queue_summary(queue: pl.DataFrame, db_path: Path = DB_PATH) -> dict:
    """Counts for the report: total invoices, clean vs needs-review."""
    con = sqlite3.connect(db_path)
    try:
        total = con.execute("SELECT COUNT(*) FROM invoices").fetchone()[0]
    finally:
        con.close()
    flagged = queue.height
    return {"total_invoices": total, "needs_review": flagged, "clean": total - flagged}
