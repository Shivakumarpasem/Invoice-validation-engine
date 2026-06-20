"""Cleaning / standardization.

Decision (Step 4): keep the ORIGINAL data untouched, add a DERIVED normalized
column next to it. So `party_name` stays exactly as received (auditors see the
real name) and `party_name_norm` is a stripped-down version the duplicate
detector compares on. Reversible, and nothing is lost.

Most other standardization is already done by typed loading (dates are real
Dates, money is exact Decimal). Here we just add the normalized name and a few
cheap text fixes.
"""

from __future__ import annotations

import polars as pl

# Legal suffixes / filler words that add noise when matching company names.
# "Acme Corp" vs "ACME Corporation Ltd" should collapse toward the same key.
_NOISE_WORDS = r"\b(ltd|limited|inc|incorporated|corp|corporation|llc|co|company|and|group|holdings)\b"


def _norm_name_expr(src: str) -> pl.Expr:
    """polars expression: company name -> normalized matching key."""
    return (
        pl.col(src)
        .str.to_lowercase()
        .str.replace_all(r"[^a-z0-9 ]", " ")   # punctuation/&/commas -> space
        .str.replace_all(_NOISE_WORDS, " ")     # drop legal suffixes & filler
        .str.replace_all(r"\s+", " ")           # collapse runs of spaces
        .str.strip_chars()                       # trim ends
    )


def clean_frames(frames: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Return a new dict of frames with normalization applied."""
    frames = dict(frames)  # shallow copy so we don't mutate the caller's dict

    frames["parties"] = frames["parties"].with_columns(
        _norm_name_expr("party_name").alias("party_name_norm")
    )

    frames["invoices"] = frames["invoices"].with_columns(
        _norm_name_expr("party_name").alias("party_name_norm"),
        pl.col("currency").str.to_uppercase(),  # tiny standardization example
    )

    return frames
