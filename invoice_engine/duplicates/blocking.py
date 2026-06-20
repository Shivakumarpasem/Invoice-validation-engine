"""Blocking - cheaply narrow 2.3M possible pairs down to candidate pairs.

Why this exists: comparing every invoice against every other is O(n^2) - ~2.3M
pairs at 2,160 invoices, and 12.5 TRILLION at a real client's 5M/year. No real
record-linkage system does that; they all "block" first: group records that
COULD match, only score within groups. This is the standard entity-resolution
move (a.k.a. indexing).

Decision (Step 6.2): block on ``party_id``. Measured against the answer key, all
160 planted duplicates still share party_id with their original, and party_id is
master data (not free text) - to change it a fraudster needs a whole second
vendor profile, which is a DIFFERENT detection vector we tackle later. So this
key keeps 100% of true pairs here while cutting cost ~40x.

Tradeoff owned: blocking sets a hard CEILING on recall - any true pair that does
NOT share the block key can never be scored. We chose the key precisely so that
ceiling stays at 100% for the duplicates we target now. A fuzzy-name blocking
branch (for same-vendor-two-profiles) is a planned, measured extension.
"""

from __future__ import annotations

from itertools import combinations

import polars as pl


def candidate_pairs(invoices: pl.DataFrame) -> pl.DataFrame:
    """Return candidate pairs ``(record_id_a, record_id_b)`` to be scored.

    Two invoices are a candidate only if they share ``party_id``. Within each
    party group we emit every unordered pair (a < b so no pair is duplicated and
    no invoice is paired with itself).
    """
    # record_ids grouped per party: {party_id: [record_id, ...]}
    groups = (
        invoices.group_by("party_id")
        .agg(pl.col("record_id"))
        .get_column("record_id")
        .to_list()
    )

    a_ids: list[int] = []
    b_ids: list[int] = []
    for record_ids in groups:
        if len(record_ids) < 2:
            continue  # a lone invoice in its party has nothing to pair with
        for a, b in combinations(sorted(record_ids), 2):
            a_ids.append(a)
            b_ids.append(b)

    return pl.DataFrame(
        {"record_id_a": a_ids, "record_id_b": b_ids},
        schema={"record_id_a": pl.Int64, "record_id_b": pl.Int64},
    )
