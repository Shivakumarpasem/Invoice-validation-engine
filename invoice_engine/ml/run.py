"""Train the ML duplicate classifier and report how well it generalizes.

Run it:  python -m invoice_engine.ml.run

Pipeline: run the detector (block + score) -> attach ground-truth labels ->
build (X features, y labels) -> train logistic regression on a TRAIN split ->
evaluate on a held-out TEST split -> show the learned weights and a side-by-side
vs the hand-tuned rule.

This module is OPTIONAL and self-contained: the core duplicate engine
(generator -> ingestion -> validation -> scoring -> reporting) runs fully without
it. This step only trains/evaluates a model on the features that engine produces.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ..duplicates.grade import true_pairs
from ..duplicates.run import detect
from ..ingestion.store import DB_PATH
from ..duplicates.persist import CONFIDENCE_THRESHOLD
from .dataset import label_pairs, to_xy
from .model import TEST_FRACTION, train_and_evaluate


def _load_truth(db_path: Path):
    # Reuse the detector's ground-truth reader (handles the null duplicate_of).
    from ..duplicates.run import _load_ground_truth
    return true_pairs(_load_ground_truth(db_path))


def run(db_path: Path = DB_PATH):
    scored = detect(db_path)
    truth = _load_truth(db_path)
    labeled = label_pairs(scored, truth)
    X, y = to_xy(labeled)
    return train_and_evaluate(X, y), labeled


def _report(db_path: Path = DB_PATH) -> None:
    result, labeled = run(db_path)

    total = labeled.height
    positives = int(labeled.get_column("label").sum())

    print("=" * 60)
    print("  ML DUPLICATE CLASSIFIER (logistic regression)")
    print("=" * 60)
    print(f"\ncandidate pairs (rows)   : {total}")
    print(f"  real duplicates (label 1): {positives}")
    print(f"  non-duplicates (label 0) : {total - positives}")
    print(f"\ntrain / test split       : "
          f"{result.n_train} train / {result.n_test} test "
          f"({int(TEST_FRACTION * 100)}% held out, {result.test_positives} duplicates in test)")

    print("\n=== EVALUATION ON THE HELD-OUT TEST SET ===")
    print("(scores on pairs the model NEVER saw during training)")
    print(f"  precision : {result.precision:.3f}")
    print(f"  recall    : {result.recall:.3f}")
    print(f"  F1        : {result.f1:.3f}")

    print("\n=== WHAT THE MODEL LEARNED (feature weights) ===")
    print("(higher weight = that signal pushes harder toward 'duplicate')")
    for name, w in sorted(result.learned_weights.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {name:12}: {w:+.3f}")
    print(f"  intercept   : {result.intercept:+.3f}")

    print("\n=== LEARNED MODEL vs HAND-TUNED RULE ===")
    print(f"  hand-tuned rule : fixed weights (.40/.20/.25/.15), "
          f"threshold {CONFIDENCE_THRESHOLD} - chosen by us.")
    print("  learned model   : weights above + boundary - learned from labeled examples,")
    print("                    and scored on data it never saw. Same inputs, learned decision.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    _report()
