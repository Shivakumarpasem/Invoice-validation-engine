"""Run the full fuzzy duplicate detector and prove it works.

Run it:  python -m invoice_engine.duplicates.run

Pipeline: load typed invoices -> block into candidate pairs -> score each pair
-> grade against the ground-truth answer key (precision/recall/F1 across
thresholds) -> show the side-by-side win over naive exact-match.

This is the centerpiece's proof. It does NOT yet persist findings to SQLite -
that table is defined and written in Step 7 (the review queue), once we know the
chosen threshold and output shape.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import polars as pl

from ..ingestion.store import DB_PATH
from ..validation.load import load_for_validation
from .blocking import candidate_pairs
from .grade import (
    exact_match_baseline,
    sweep_thresholds,
    true_pairs,
)
from .score import score_pairs


def _load_ground_truth(db_path: Path) -> pl.DataFrame:
    con = sqlite3.connect(db_path)
    try:
        cur = con.execute(
            "SELECT record_id, problem_family, problem_type, duplicate_of FROM ground_truth"
        )
        # Explicit schema: duplicate_of is null for integrity rows, so letting
        # polars infer from the first (null) rows would lock it to Null and then
        # fail when a real int appears further down.
        schema = {
            "record_id": pl.Int64,
            "problem_family": pl.String,
            "problem_type": pl.String,
            "duplicate_of": pl.Int64,
        }
        return pl.DataFrame(cur.fetchall(), schema=schema, orient="row")
    finally:
        con.close()


def detect(db_path: Path = DB_PATH) -> pl.DataFrame:
    """Block + score every candidate pair. Returns the scored pairs."""
    invoices, _ = load_for_validation(db_path)
    pairs = candidate_pairs(invoices)
    return score_pairs(invoices, pairs)


def _report(db_path: Path = DB_PATH) -> None:
    invoices, _ = load_for_validation(db_path)
    ground_truth = _load_ground_truth(db_path)
    truth = true_pairs(ground_truth)

    pairs = candidate_pairs(invoices)
    scored = score_pairs(invoices, pairs)

    print("=== DUPLICATE DETECTOR ===")
    print(f"invoices            : {invoices.height}")
    print(f"candidate pairs      : {pairs.height}  (after party_id blocking)")
    print(f"real duplicate pairs : {len(truth)}  (the answer key)\n")

    print("=== PRECISION / RECALL ACROSS THRESHOLDS ===")
    sweep = sweep_thresholds(scored, truth)
    with pl.Config(tbl_width_chars=200, tbl_rows=20):
        print(sweep)

    best = sweep.sort("f1", descending=True).head(1).to_dicts()[0]
    print(
        f"\nBest F1 at threshold {best['threshold']}: "
        f"precision {best['precision']}, recall {best['recall']}, F1 {best['f1']} "
        f"({best['true_positives']} caught, {best['false_positives']} false alarms, "
        f"{best['missed']} missed)"
    )

    print("\n=== SIDE BY SIDE: fuzzy vs naive exact-match ===")
    baseline = exact_match_baseline(invoices, truth)
    print(f"  naive exact-match : caught {baseline['true_positives']}/{len(truth)} "
          f"(recall {baseline['recall']}), precision {baseline['precision']}")
    print(f"  fuzzy @ {best['threshold']}      : caught {best['true_positives']}/{len(truth)} "
          f"(recall {best['recall']}), precision {best['precision']}")
    gained = best["true_positives"] - baseline["true_positives"]
    print(f"  -> fuzzy catches {gained} near-duplicates exact-match misses")

    print("\n=== WHAT EXACT-MATCH MISSES, BY TRICK TYPE ===")
    _breakdown_by_type(scored, ground_truth, threshold=best["threshold"])


def _breakdown_by_type(scored: pl.DataFrame, ground_truth: pl.DataFrame, threshold: float) -> None:
    """For each planted trick type, how many did the fuzzy detector catch?"""
    flagged = {
        frozenset((r["record_id_a"], r["record_id_b"]))
        for r in scored.filter(pl.col("confidence") >= threshold).iter_rows(named=True)
    }
    dupes = ground_truth.filter(pl.col("problem_family") == "duplicate")
    rows = []
    for ptype in ["exact_duplicate", "leading_zero", "reworded_vendor", "tweaked_number"]:
        sub = dupes.filter(pl.col("problem_type") == ptype)
        planted = sub.height
        caught = sum(
            1
            for r in sub.iter_rows(named=True)
            if frozenset((r["record_id"], r["duplicate_of"])) in flagged
        )
        rows.append({"trick": ptype, "planted": planted, "caught": caught})
    with pl.Config(tbl_width_chars=200):
        print(pl.DataFrame(rows))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    _report()
