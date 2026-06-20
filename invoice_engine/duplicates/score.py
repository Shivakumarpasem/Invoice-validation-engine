"""Score candidate pairs - blend per-field signals into one confidence number.

For each candidate pair (from blocking) we compute four similarity signals, then
combine them with a weighted average into a single ``confidence`` in [0, 1].

Why a weighted blend (not a single field): no one field catches every trick.
  * leading_zero / tweaked_number  -> id_sim carries it (name/amount/date equal).
  * reworded_vendor                -> id changed AND name changed; neither alone
    is decisive, but amount + date still match - the blend sees the whole picture.
A single confidence number also gives Step 7 one thing to rank the review queue by.

The four signals:
  * id_sim     - invoice_id character edit-distance (formatting tricks).
  * name_sim   - vendor name token-set overlap (rewording).
  * amount_sim - exact match -> 1.0 else 0.0. Money is the strongest "same bill"
    tell; a different amount is almost always a different bill, so we treat it as
    a hard 0/1 rather than a fuzzy ratio.
  * date_sim   - same date -> 1.0, decaying as the dates drift apart (a genuine
    resubmission is usually days, not months, apart).

Weights are deliberately simple defaults here; we TUNE them against
precision/recall in Step 6.3 rather than freezing a guess now.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from .similarity import invoice_id_similarity, name_similarity

# Default blend weights (sum to 1.0). Tuned in 6.3 against the answer key.
WEIGHTS = {
    "id_sim": 0.40,      # the headline formatting-forgery signal
    "name_sim": 0.20,    # catches rewording; usually ~1.0 inside a party block
    "amount_sim": 0.25,  # money equality - strong "same bill" evidence
    "date_sim": 0.15,    # timing proximity - supporting evidence
}

# Date proximity: identical = 1.0, linearly decaying to 0 at this many days apart.
_DATE_DECAY_DAYS = 30


def _amount_sim(a: str, b: str) -> float:
    """Exact decimal-string equality -> 1.0, else 0.0 (money is binary evidence)."""
    return 1.0 if a == b else 0.0


def _date_sim(a: date | None, b: date | None) -> float:
    """1.0 if same day, decaying linearly to 0.0 at _DATE_DECAY_DAYS apart."""
    if a is None or b is None:
        return 0.0
    gap = abs((a - b).days)
    if gap >= _DATE_DECAY_DAYS:
        return 0.0
    return 1.0 - gap / _DATE_DECAY_DAYS


def _confidence(id_sim: float, name_sim: float, amount_sim: float, date_sim: float) -> float:
    """Weighted average of the four signals -> one confidence in [0, 1]."""
    return (
        WEIGHTS["id_sim"] * id_sim
        + WEIGHTS["name_sim"] * name_sim
        + WEIGHTS["amount_sim"] * amount_sim
        + WEIGHTS["date_sim"] * date_sim
    )


def score_pairs(invoices: pl.DataFrame, pairs: pl.DataFrame) -> pl.DataFrame:
    """Score every candidate pair. Returns the pairs with per-signal columns and
    a final ``confidence``, sorted most-suspicious first.

    ``invoices`` must carry typed columns (invoice_date as Date, amounts as the
    exact text we stored) - we look fields up by record_id.
    """
    # Index invoice fields by record_id for O(1) lookup while iterating pairs.
    by_id = {
        row["record_id"]: row
        for row in invoices.select(
            "record_id", "invoice_id", "party_name_norm", "invoice_amount", "invoice_date"
        ).iter_rows(named=True)
    }

    rows: list[dict] = []
    for a_id, b_id in pairs.iter_rows():
        a, b = by_id[a_id], by_id[b_id]
        id_sim = invoice_id_similarity(a["invoice_id"], b["invoice_id"])
        name_sim = name_similarity(a["party_name_norm"], b["party_name_norm"])
        amount_sim = _amount_sim(str(a["invoice_amount"]), str(b["invoice_amount"]))
        date_sim = _date_sim(a["invoice_date"], b["invoice_date"])
        rows.append(
            {
                "record_id_a": a_id,
                "record_id_b": b_id,
                "id_sim": round(id_sim, 4),
                "name_sim": round(name_sim, 4),
                "amount_sim": amount_sim,
                "date_sim": round(date_sim, 4),
                "confidence": round(
                    _confidence(id_sim, name_sim, amount_sim, date_sim), 4
                ),
            }
        )

    schema = {
        "record_id_a": pl.Int64,
        "record_id_b": pl.Int64,
        "id_sim": pl.Float64,
        "name_sim": pl.Float64,
        "amount_sim": pl.Float64,
        "date_sim": pl.Float64,
        "confidence": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema).sort("confidence", descending=True)
