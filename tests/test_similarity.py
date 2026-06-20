"""Tests for the per-field similarity metrics (the hybrid core).

These prove the two metrics behave the way the whole detector depends on:
near-identical forgeries score HIGH, unrelated values score LOW.
"""

from invoice_engine.duplicates.similarity import (
    invoice_id_similarity,
    name_similarity,
)


def test_identical_invoice_ids_score_one():
    assert invoice_id_similarity("412809", "412809") == 1.0


def test_leading_zero_scores_high():
    # The headline trick: a single inserted leading zero must stay near 1.0.
    assert invoice_id_similarity("412809", "0412809") > 0.85


def test_tweaked_number_scores_high():
    # One appended suffix is still a near-duplicate.
    assert invoice_id_similarity("371246", "371246-A") > 0.80


def test_unrelated_invoice_ids_score_low():
    assert invoice_id_similarity("412809", "998001") < 0.5


def test_empty_invoice_id_scores_zero():
    assert invoice_id_similarity("", "412809") == 0.0


def test_reworded_name_scores_high_after_normalization():
    # Normalized names: token-set overlap should keep the shared core high even
    # when extra words are added.
    assert name_similarity("acme", "acme corporation holdings") == 1.0


def test_identical_names_score_one():
    assert name_similarity("morales jones", "morales jones") == 1.0


def test_unrelated_names_score_low():
    assert name_similarity("perez", "morales jones") < 0.5
