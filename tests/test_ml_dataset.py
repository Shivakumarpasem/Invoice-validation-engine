"""Tests for the ML feature/label table builder.

These prove the bridge from the detector to the model is correct: a pair in the
answer key gets label 1, others get 0, and the (X, y) arrays have the right shape.
"""

import polars as pl

from invoice_engine.ml.dataset import FEATURE_COLUMNS, label_pairs, to_xy


def _scored(rows):
    schema = {
        "record_id_a": pl.Int64,
        "record_id_b": pl.Int64,
        "id_sim": pl.Float64,
        "name_sim": pl.Float64,
        "amount_sim": pl.Float64,
        "date_sim": pl.Float64,
        "confidence": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema, orient="row")


def test_label_is_one_for_true_pairs_zero_otherwise():
    scored = _scored([
        (1, 2, 1.0, 1.0, 1.0, 1.0, 1.0),   # this pair IS in the answer key
        (3, 4, 0.2, 0.3, 0.0, 0.1, 0.15),  # this one is not
    ])
    truth = {frozenset((1, 2))}
    labeled = label_pairs(scored, truth)
    labels = dict(
        (frozenset((r["record_id_a"], r["record_id_b"])), r["label"])
        for r in labeled.iter_rows(named=True)
    )
    assert labels[frozenset((1, 2))] == 1
    assert labels[frozenset((3, 4))] == 0


def test_label_is_order_independent():
    # Answer key stores {fake, original}; the scored pair may be in either order.
    scored = _scored([(9, 5, 1.0, 1.0, 1.0, 1.0, 1.0)])
    truth = {frozenset((5, 9))}
    assert label_pairs(scored, truth).get_column("label").to_list() == [1]


def test_to_xy_shapes_and_features():
    scored = _scored([
        (1, 2, 1.0, 1.0, 1.0, 1.0, 1.0),
        (3, 4, 0.2, 0.3, 0.0, 0.1, 0.15),
    ])
    labeled = label_pairs(scored, {frozenset((1, 2))})
    X, y = to_xy(labeled)
    assert X.shape == (2, len(FEATURE_COLUMNS))  # 2 rows, 4 features
    assert y.tolist() == [1, 0]
