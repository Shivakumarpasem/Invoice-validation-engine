"""Tests for blocking - the candidate-pair generation.

Blocking sets the recall ceiling, so these prove it does exactly what we claim:
only pairs that SHARE party_id become candidates, every such pair appears once,
and single-invoice parties produce nothing.
"""

import polars as pl

from invoice_engine.duplicates.blocking import candidate_pairs


def _invoices(rows):
    return pl.DataFrame(
        rows, schema={"record_id": pl.Int64, "party_id": pl.String}, orient="row"
    )


def test_only_same_party_pairs_are_candidates():
    # Two invoices for V1, one for V2. Only the V1 pair should appear.
    inv = _invoices([(1, "V1"), (2, "V1"), (3, "V2")])
    pairs = candidate_pairs(inv)
    got = {frozenset(p) for p in pairs.iter_rows()}
    assert got == {frozenset((1, 2))}


def test_pairs_within_a_party_are_complete():
    # 3 invoices in one party -> C(3,2) = 3 pairs.
    inv = _invoices([(1, "V1"), (2, "V1"), (3, "V1")])
    pairs = candidate_pairs(inv)
    got = {frozenset(p) for p in pairs.iter_rows()}
    assert got == {frozenset((1, 2)), frozenset((1, 3)), frozenset((2, 3))}


def test_single_invoice_party_yields_no_pairs():
    inv = _invoices([(1, "V1"), (2, "V2"), (3, "V3")])
    assert candidate_pairs(inv).height == 0


def test_pairs_are_ordered_a_less_than_b():
    inv = _invoices([(5, "V1"), (2, "V1")])
    a, b = candidate_pairs(inv).row(0)
    assert a < b
