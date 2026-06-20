"""Tests for the signal blend - score_pairs.

These prove the confidence number is trustworthy: an exact duplicate pins at 1.0,
an unrelated pair scores low, and the date-decay helper behaves.
"""

from datetime import date

import polars as pl

from invoice_engine.duplicates.score import _date_sim, score_pairs


def _invoices(rows):
    return pl.DataFrame(
        rows,
        schema={
            "record_id": pl.Int64,
            "invoice_id": pl.String,
            "party_name_norm": pl.String,
            "invoice_amount": pl.String,
            "invoice_date": pl.Date,
        },
        orient="row",
    )


def _pairs(rows):
    return pl.DataFrame(
        rows, schema={"record_id_a": pl.Int64, "record_id_b": pl.Int64}, orient="row"
    )


def test_exact_duplicate_scores_confidence_one():
    d = date(2026, 1, 1)
    inv = _invoices([
        (1, "412809", "acme", "100.00", d),
        (2, "412809", "acme", "100.00", d),
    ])
    scored = score_pairs(inv, _pairs([(1, 2)]))
    assert scored.row(0, named=True)["confidence"] == 1.0


def test_unrelated_pair_scores_low():
    inv = _invoices([
        (1, "412809", "acme", "100.00", date(2026, 1, 1)),
        (2, "998001", "globex", "9999.00", date(2025, 1, 1)),
    ])
    scored = score_pairs(inv, _pairs([(1, 2)]))
    assert scored.row(0, named=True)["confidence"] < 0.5


def test_leading_zero_pair_clears_threshold():
    # The forgery the project exists to catch should score as a near-duplicate.
    d = date(2026, 1, 1)
    inv = _invoices([
        (1, "412809", "acme", "100.00", d),
        (2, "0412809", "acme", "100.00", d),
    ])
    scored = score_pairs(inv, _pairs([(1, 2)]))
    assert scored.row(0, named=True)["confidence"] >= 0.90


def test_date_sim_same_day_is_one():
    assert _date_sim(date(2026, 1, 1), date(2026, 1, 1)) == 1.0


def test_date_sim_far_apart_is_zero():
    assert _date_sim(date(2026, 1, 1), date(2026, 6, 1)) == 0.0


def test_date_sim_handles_null():
    assert _date_sim(None, date(2026, 1, 1)) == 0.0
