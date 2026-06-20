"""Build the (features, labels) table a classifier trains on.

This is the bridge from the similarity engine to real ML. The hard part of any
ML project - turning raw records into a clean feature matrix with labels - is
already done by blocking + scoring; this module just assembles it:

  * FEATURES (the clues, X): the 4 per-pair similarity signals the scorer already
    computes - id_sim, name_sim, amount_sim, date_sim.
  * LABEL (the answer, y): 1 if this pair is a real duplicate (it appears in the
    ground-truth answer key), else 0.

So each candidate pair becomes one training row: [4 features] -> [0 or 1].
"""

from __future__ import annotations

import polars as pl

FEATURE_COLUMNS = ["id_sim", "name_sim", "amount_sim", "date_sim"]


def label_pairs(scored: pl.DataFrame, truth: set[frozenset[int]]) -> pl.DataFrame:
    """Attach the ground-truth label to every scored candidate pair.

    ``truth`` is the answer key as a set of unordered {fake, original} pairs
    (from ``duplicates.grade.true_pairs``). A scored pair gets label 1 if it is
    in that set, else 0.
    """
    labels = [
        1 if frozenset((a, b)) in truth else 0
        for a, b in scored.select("record_id_a", "record_id_b").iter_rows()
    ]
    return scored.with_columns(pl.Series("label", labels, dtype=pl.Int64))


def to_xy(labeled: pl.DataFrame):
    """Split the labeled frame into ``(X, y)`` numpy arrays for scikit-learn.

    X = the 4 feature columns (n_pairs x 4); y = the 0/1 label per pair. This is
    the polars -> numpy seam noted back in the stack decision: ML libraries want
    numpy, so we convert at the boundary.
    """
    X = labeled.select(FEATURE_COLUMNS).to_numpy()
    y = labeled.get_column("label").to_numpy()
    return X, y
