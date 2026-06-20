"""Grade the detector against the ground-truth answer key.

This is the PROOF step: it turns the scored candidate pairs into precision /
recall / F1 numbers, so we can state - with evidence - how well the fuzzy
detector catches the planted duplicates, and show it beating naive exact-match.

Two beginner-level definitions, because they ARE this module:
  * recall    = of all the REAL duplicate pairs, what fraction did we catch?
                (low recall = we missed real duplicates)
  * precision = of the pairs we FLAGGED, what fraction were really duplicates?
                (low precision = we raised false alarms)
  * F1        = the harmonic mean of the two - one number that is only high when
                BOTH are high (so you can't game it by flagging everything).

Grading rule (Step 6.3 decision): PAIR-LEVEL, strict. A flagged pair counts as a
true catch only if that exact (fake, original) pair is in the answer key, in
either order. The most honest record-linkage evaluation; if it penalises an
arguably-right transitive pair, that only UNDERstates our precision - the safe
direction.
"""

from __future__ import annotations

import polars as pl


def true_pairs(ground_truth: pl.DataFrame) -> set[frozenset[int]]:
    """The answer key as a set of unordered {fake, original} pairs.

    Each duplicate row links a fake record_id to the original it copies
    (``duplicate_of``). We store each as a frozenset so pair order never matters
    when we compare against our scored pairs.
    """
    dupes = ground_truth.filter(pl.col("problem_family") == "duplicate")
    return {
        frozenset((row["record_id"], row["duplicate_of"]))
        for row in dupes.iter_rows(named=True)
        if row["duplicate_of"] is not None
    }


def _metrics_at(scored: pl.DataFrame, truth: set[frozenset[int]], threshold: float) -> dict:
    """Precision / recall / F1 if we flag every pair with confidence >= threshold."""
    flagged = {
        frozenset((row["record_id_a"], row["record_id_b"]))
        for row in scored.filter(pl.col("confidence") >= threshold).iter_rows(named=True)
    }
    true_positives = len(flagged & truth)   # flagged AND really a duplicate
    predicted = len(flagged)                # everything we flagged
    actual = len(truth)                     # all real duplicate pairs

    precision = true_positives / predicted if predicted else 0.0
    recall = true_positives / actual if actual else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "threshold": threshold,
        "flagged": predicted,
        "true_positives": true_positives,
        "false_positives": predicted - true_positives,
        "missed": actual - true_positives,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def sweep_thresholds(
    scored: pl.DataFrame,
    truth: set[frozenset[int]],
    thresholds: list[float] | None = None,
) -> pl.DataFrame:
    """Compute the metrics across a range of thresholds, best F1 first sortable."""
    thresholds = thresholds or [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50]
    rows = [_metrics_at(scored, truth, t) for t in thresholds]
    return pl.DataFrame(rows)


def exact_match_baseline(invoices: pl.DataFrame, truth: set[frozenset[int]]) -> dict:
    """The naive ERP check, for the side-by-side comparison.

    Exact-match flags two invoices as duplicates only if (party_id, invoice_id,
    invoice_amount) are byte-for-byte identical. This is what beats it: a single
    leading zero or one tweaked digit makes invoice_id differ, so every
    formatting forgery slips through.
    """
    seen: dict[tuple, list[int]] = {}
    for row in invoices.iter_rows(named=True):
        key = (row["party_id"], row["invoice_id"], str(row["invoice_amount"]))
        seen.setdefault(key, []).append(row["record_id"])

    # Any key with >1 record => those records are flagged as exact duplicates.
    flagged: set[frozenset[int]] = set()
    for record_ids in seen.values():
        if len(record_ids) > 1:
            for i in range(len(record_ids)):
                for j in range(i + 1, len(record_ids)):
                    flagged.add(frozenset((record_ids[i], record_ids[j])))

    true_positives = len(flagged & truth)
    precision = true_positives / len(flagged) if flagged else 0.0
    recall = true_positives / len(truth) if truth else 0.0
    return {
        "method": "exact_match (naive ERP)",
        "flagged": len(flagged),
        "true_positives": true_positives,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
    }
