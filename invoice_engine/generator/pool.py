"""The party master pool.

Real AP/AR data has a *fixed set of vendors/customers that recur* across many
invoices. We generate that pool ONCE (with Faker) and every invoice draws from
it. This recurrence is what makes near-duplicate detection meaningful later:
the same real party showing up many times is exactly where reworded-name and
double-profile tricks live.
"""

from __future__ import annotations

import random

from faker import Faker
import polars as pl

from .config import GeneratorConfig

# Small, realistic option sets. Real ERPs draw these from master data, so a
# fixed vocabulary (not random noise) is the honest model.
_PAYMENT_TERMS = ["Net 15", "Net 30", "Net 45", "Net 60", "Due on receipt"]
_GL_CODES = ["5000", "5100", "5200", "6000", "6100", "6200"]
_COST_CENTERS = ["CC-100", "CC-200", "CC-300", "CC-400"]


def build_party_pool(config: GeneratorConfig) -> pl.DataFrame:
    """Return the fixed pool of vendors (AP) and customers (AR).

    Columns: party_id, party_name, direction, default_payment_terms,
    gl_account_code, cost_center.
    """
    fake = Faker()
    Faker.seed(config.seed)
    rng = random.Random(config.seed)

    n_ap = round(config.n_parties * config.ap_ratio)
    rows: list[dict] = []

    for i in range(config.n_parties):
        is_ap = i < n_ap
        direction = "AP" if is_ap else "AR"
        # V#### for vendors (AP), C#### for customers (AR)
        prefix = "V" if is_ap else "C"
        seq = (i + 1) if is_ap else (i - n_ap + 1)
        rows.append(
            {
                "party_id": f"{prefix}{seq:04d}",
                "party_name": fake.company(),
                "direction": direction,
                "default_payment_terms": rng.choice(_PAYMENT_TERMS),
                "gl_account_code": rng.choice(_GL_CODES),
                "cost_center": rng.choice(_COST_CENTERS),
            }
        )

    return pl.DataFrame(rows)
