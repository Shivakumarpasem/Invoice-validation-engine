"""Render the engine's results as a sectioned CLI report.

This is the human-facing view - the thing a reviewer (or interviewer) reads to
understand what the engine did. It does no detection itself; it reads the
already-computed queue + findings and formats them.

Four sections, briefing-first:
  1. SUMMARY        - totals, needs-review, % auto-approvable.
  2. THE HEADLINE   - fuzzy detector vs naive exact-match (the ML win).
  3. BREAKDOWN      - problems by type (duplicate tricks + validation checks).
  4. TOP CASES      - the worst N invoices with their reasons.

The full worklist is available on demand (--all / --csv) rather than flooding
the terminal, so the briefing stays readable.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import polars as pl

from ..duplicates.grade import exact_match_baseline, true_pairs
from ..duplicates.persist import CONFIDENCE_THRESHOLD
from ..ingestion.store import DB_PATH
from ..scoring.queue import build_queue, queue_summary

_RULE = "=" * 64


def _load_ground_truth(con: sqlite3.Connection) -> pl.DataFrame:
    cur = con.execute(
        "SELECT record_id, problem_family, problem_type, duplicate_of FROM ground_truth"
    )
    schema = {
        "record_id": pl.Int64,
        "problem_family": pl.String,
        "problem_type": pl.String,
        "duplicate_of": pl.Int64,
    }
    return pl.DataFrame(cur.fetchall(), schema=schema, orient="row")


def _section(title: str) -> None:
    print(f"\n{_RULE}\n  {title}\n{_RULE}")


def _summary(queue: pl.DataFrame, db_path: Path) -> None:
    s = queue_summary(queue, db_path)
    _section("1. SUMMARY")
    print(f"  invoices ingested   : {s['total_invoices']}")
    print(f"  needs review        : {s['needs_review']}")
    print(f"  clean / approvable  : {s['clean']}  "
          f"({100 * s['clean'] / s['total_invoices']:.1f}%)")


def _headline(db_path: Path) -> None:
    """The ML win: fuzzy detector vs the naive exact-match an ERP would use."""
    con = sqlite3.connect(db_path)
    try:
        invoices = _read(con, "invoices")
        truth = true_pairs(_load_ground_truth(con))
        dup_caught = con.execute("SELECT COUNT(*) FROM duplicate_findings").fetchone()[0]
    finally:
        con.close()

    baseline = exact_match_baseline(invoices, truth)
    total_real = len(truth)

    _section("2. THE HEADLINE - fuzzy vs naive exact-match")
    print(f"  real duplicate pairs planted : {total_real}")
    print(f"  naive exact-match catches    : {baseline['true_positives']:>3}  "
          f"(recall {baseline['recall']:.0%})")
    print(f"  fuzzy detector catches       : {dup_caught:>3}  "
          f"(recall {dup_caught / total_real:.0%})  @ confidence >= {CONFIDENCE_THRESHOLD}")
    print(f"  -> fuzzy recovers {dup_caught - baseline['true_positives']} "
          f"near-duplicates that exact-match silently misses.")


def _breakdown(db_path: Path) -> None:
    con = sqlite3.connect(db_path)
    try:
        gt = _load_ground_truth(con)
        flagged_pairs = {
            frozenset((a, b))
            for a, b in con.execute(
                "SELECT record_id_a, record_id_b FROM duplicate_findings"
            ).fetchall()
        }
        val = _read(con, "validation_findings")
    finally:
        con.close()

    _section("3. BREAKDOWN BY PROBLEM TYPE")

    print("  Duplicates (caught / planted, by trick):")
    dupes = gt.filter(pl.col("problem_family") == "duplicate")
    for ptype in ["exact_duplicate", "leading_zero", "reworded_vendor", "tweaked_number"]:
        sub = dupes.filter(pl.col("problem_type") == ptype)
        planted = sub.height
        caught = sum(
            1 for r in sub.iter_rows(named=True)
            if frozenset((r["record_id"], r["duplicate_of"])) in flagged_pairs
        )
        print(f"    {ptype:18}: {caught:>3} / {planted}")

    print("\n  Validation findings (by check):")
    if val.is_empty():
        print("    (none)")
    else:
        by_check = val.group_by("check_name", "severity").len().sort("check_name")
        for r in by_check.iter_rows(named=True):
            print(f"    {r['check_name']:16} [{r['severity']:7}]: {r['len']}")


def _top_cases(queue: pl.DataFrame, n: int) -> None:
    _section(f"4. TOP {n} CASES TO REVIEW (worst first)")
    top = queue.head(n)
    for i, row in enumerate(top.iter_rows(named=True), start=1):
        amt = row["invoice_amount"]
        print(f"  {i:>2}. [priority {row['priority']:.2f}]  "
              f"invoice {row['invoice_id']}  {row['party_name']}  (${amt})")
        for reason in row["reasons"].split(" | "):
            print(f"        - {reason}")


def _read(con: sqlite3.Connection, name: str) -> pl.DataFrame:
    cur = con.execute(f"SELECT * FROM {name}")
    cols = [d[0] for d in cur.description]
    return pl.DataFrame(cur.fetchall(), schema=cols, orient="row")


def render_report(
    db_path: Path = DB_PATH,
    top_n: int = 10,
    show_all: bool = False,
    csv_path: Path | None = None,
) -> pl.DataFrame:
    """Print the sectioned report. Returns the full queue (for --all / --csv)."""
    queue = build_queue(db_path)

    print(_RULE)
    print("  INVOICE VALIDATION ENGINE - REVIEW REPORT")
    print(_RULE)

    _summary(queue, db_path)
    _headline(db_path)
    _breakdown(db_path)
    _top_cases(queue, top_n)

    if show_all:
        _section(f"FULL WORKLIST ({queue.height} invoices)")
        with pl.Config(fmt_str_lengths=100, tbl_width_chars=220, tbl_rows=-1):
            print(queue.select(
                "record_id", "invoice_id", "party_name", "priority", "reasons"
            ))

    if csv_path is not None:
        queue.write_csv(csv_path)
        print(f"\n  Full worklist written to: {csv_path}")

    print()
    return queue
