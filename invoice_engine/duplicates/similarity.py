"""Per-field similarity metrics - the hybrid core of the duplicate detector.

Decision (Step 6): different fields get forged differently, so each gets the
metric that matches its failure mode. Both come from one library (rapidfuzz),
so the hybrid costs us no extra dependency.

  * invoice_id   -> EDIT-DISTANCE (character level). The forgeries here are
    formatting tricks: "00123" vs "123" (one inserted char), "371246" vs
    "371246-A" (two appended chars), one digit flipped. Character edit-distance
    measures exactly that: how many single-character edits separate the two.

  * party_name   -> TOKEN-SET ratio (word level). Vendor rewording reorders or
    adds words: "Acme Corp" vs "ACME Corporation Ltd". Character distance
    punishes that harshly; comparing the SET of words ignores order and extra
    words, so the shared core ("acme") still scores high.

Every function returns a similarity in [0.0, 1.0] where 1.0 == identical. We
divide rapidfuzz's 0-100 scores by 100 so the whole detector speaks one scale.
"""

from __future__ import annotations

from rapidfuzz import fuzz


def invoice_id_similarity(a: str, b: str) -> float:
    """Character edit-distance similarity for the printed invoice number.

    rapidfuzz's ``ratio`` is built on Levenshtein (insert/delete/substitute):
    the closer the character sequences, the higher the score. A single inserted
    leading zero or one appended suffix barely moves it off 1.0 - which is the
    whole point: those near-identical numbers SHOULD score as near-duplicates.
    """
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0


def name_similarity(a_norm: str, b_norm: str) -> float:
    """Token-set similarity for the (already normalized) vendor name.

    Operates on ``party_name_norm`` (lowercased, punctuation + legal suffixes
    stripped in Step 4). ``token_set_ratio`` splits each string into a SET of
    words and compares the overlap, so word order and extra words don't tank the
    score - the right behaviour for reworded vendors.
    """
    if not a_norm or not b_norm:
        return 0.0
    return fuzz.token_set_ratio(a_norm, b_norm) / 100.0
