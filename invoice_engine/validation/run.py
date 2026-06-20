"""Run validation and store the findings.

Run it:  python -m invoice_engine.validation.run

Loads the DB -> runs all checks -> writes a `validation_findings` table back to
SQLite (this feeds the Step 7 review queue) -> prints a summary and sanity-checks
the integrity findings against the ground-truth answer key.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import polars as pl

from ..ingestion.store import DB_PATH
from .checks import run_all_checks
from .load import load_for_validation

_FINDINGS_TABLE_SQL = """
DROP TABLE IF EXISTS validation_findings;
CREATE TABLE validation_findings (
    record_id  INTEGER,
    check_name TEXT,
    severity   TEXT,
    detail     TEXT
);
"""


def _store_findings(findings: pl.DataFrame, db_path: Path) -> None:
    con = sqlite3.connect(db_path)
    try:
        con.executescript(_FINDINGS_TABLE_SQL)
        con.executemany(
            "INSERT INTO validation_findings (record_id, check_name, severity, detail) "
            "VALUES (?, ?, ?, ?)",
            findings.iter_rows(),
        )
        con.commit()
    finally:
        con.close()


def run(db_path: Path = DB_PATH) -> pl.DataFrame:
    invoices, line_items = load_for_validation(db_path)
    findings = run_all_checks(invoices, line_items)
    _store_findings(findings, db_path)
    return findings


def _summary(findings: pl.DataFrame, db_path: Path) -> None:
    print("=== VALIDATION FINDINGS ===")
    print(f"total findings: {findings.height}")
    print(f"invoices flagged (unique): {findings['record_id'].n_unique()}\n")

    by_check = findings.group_by("check_name", "severity").len().sort("check_name")
    print(by_check)

    # Cross-check integrity findings vs the planted ground truth.
    con = sqlite3.connect(db_path)
    try:
        gt = dict(
            con.execute(
                "SELECT problem_type, COUNT(*) FROM ground_truth GROUP BY problem_type"
            ).fetchall()
        )
    finally:
        con.close()
    print("\n=== SANITY CHECK vs ANSWER KEY ===")
    print(f"  broken_sum : planted {gt.get('broken_sum')} | "
          f"caught by line_item_sum {findings.filter(pl.col('check_name')=='line_item_sum').height}")
    print(f"  bad_date   : planted {gt.get('bad_date')} | "
          f"caught by date_sanity {findings.filter(pl.col('check_name')=='date_sanity').height}")
    print(f"  missing_po : planted {gt.get('missing_po')} | "
          f"flagged by missing_po {findings.filter(pl.col('check_name')=='missing_po').height} "
          f"(includes invoices that legitimately never had a PO)")

    print("\n=== SAMPLE FINDINGS ===")
    with pl.Config(fmt_str_lengths=80, tbl_width_chars=200):
        print(findings.head(6))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    findings = run()
    _summary(findings, DB_PATH)
