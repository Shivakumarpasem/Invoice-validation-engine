"""Clean invoice generation.

Produces *correct* invoices drawn from the fixed party pool: line items that
sum exactly to the header total, valid dates, consistent terms. The deliberate
problems (duplicates, broken sums, ...) are layered on top in Step 3b - keeping
"clean" and "messy" separate means our ground-truth answer key is unambiguous.

Money is handled as Python ``Decimal`` so sums are exact (no float drift), then
stored in polars as a Decimal column.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from decimal import Decimal

import polars as pl

from .config import GeneratorConfig

# Fixed reference "today" so the dataset is fully reproducible.
_REFERENCE_DATE = date(2026, 6, 1)
_CENT = Decimal("0.01")

_STATUS_UNPAID = ["pending", "approved"]

_TERMS_DAYS = {
    "Net 15": 15,
    "Net 30": 30,
    "Net 45": 45,
    "Net 60": 60,
    "Due on receipt": 0,
}

# Money/decimal columns, declared once so every builder agrees on the dtype.
_INVOICE_SCHEMA = {
    "record_id": pl.Int64,  # internal surrogate key - ALWAYS unique, links lines
    "invoice_id": pl.String,  # printed business number - allowed to repeat / be wrong
    "direction": pl.String,
    "party_id": pl.String,
    "party_name": pl.String,
    "invoice_date": pl.Date,
    "due_date": pl.Date,
    "payment_date": pl.Date,
    "invoice_amount": pl.Decimal(scale=2),
    "tax_amount": pl.Decimal(scale=2),
    "currency": pl.String,
    "payment_terms": pl.String,
    "po_number": pl.String,
    "status": pl.String,
    "gl_account_code": pl.String,
    "cost_center": pl.String,
}

_LINE_ITEM_SCHEMA = {
    "record_id": pl.Int64,  # FK -> invoices.record_id (NOT invoice_id, which can repeat)
    "description": pl.String,
    "qty": pl.Int64,
    "unit_price": pl.Decimal(scale=2),
    "line_total": pl.Decimal(scale=2),
}


def _money(value: Decimal) -> Decimal:
    """Round to cents - the only place money precision is decided."""
    return value.quantize(_CENT)


def generate_clean_invoices(
    config: GeneratorConfig, pool: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build ``(invoices_df, line_items_df)`` of fully valid invoices."""
    rng = random.Random(config.seed + 1)  # offset so it differs from pool's rng
    tax_rate = Decimal(config.tax_rate)
    parties = pool.to_dicts()

    used_ids: set[str] = set()
    invoices: list[dict] = []
    line_items: list[dict] = []

    for record_id in range(1, config.n_invoices + 1):  # 1, 2, 3 ... unique key
        party = rng.choice(parties)

        invoice_id = _unique_invoice_id(rng, used_ids)
        inv_date = _REFERENCE_DATE - timedelta(days=rng.randint(0, 730))
        terms = party["default_payment_terms"]
        due_date = inv_date + timedelta(days=_TERMS_DAYS[terms])

        # --- line items: sum exactly to the header total ---
        n_lines = rng.randint(config.min_lines, config.max_lines)
        subtotal = Decimal("0.00")
        for _ in range(n_lines):
            qty = rng.randint(1, 20)
            unit_price = _money(Decimal(rng.randint(500, 50_000)) / 100)  # $5-$500
            line_total = _money(unit_price * qty)
            subtotal += line_total
            line_items.append(
                {
                    "record_id": record_id,
                    "description": _item_description(rng),
                    "qty": qty,
                    "unit_price": unit_price,
                    "line_total": line_total,
                }
            )

        invoice_amount = _money(subtotal)
        tax_amount = _money(invoice_amount * tax_rate)

        # --- payment status / date ---
        if rng.random() < config.p_paid:
            status = "paid"
            pay_offset = rng.randint(0, _TERMS_DAYS[terms] + 15)
            payment_date = min(inv_date + timedelta(days=pay_offset), _REFERENCE_DATE)
        else:
            status = rng.choice(_STATUS_UNPAID)
            payment_date = None

        po_number = (
            f"PO{rng.randint(0, 999_999):06d}" if rng.random() < config.p_has_po else None
        )

        invoices.append(
            {
                "record_id": record_id,
                "invoice_id": invoice_id,
                "direction": party["direction"],
                "party_id": party["party_id"],
                "party_name": party["party_name"],
                "invoice_date": inv_date,
                "due_date": due_date,
                "payment_date": payment_date,
                "invoice_amount": invoice_amount,
                "tax_amount": tax_amount,
                "currency": config.currency,
                "payment_terms": terms,
                "po_number": po_number,
                "status": status,
                "gl_account_code": party["gl_account_code"],
                "cost_center": party["cost_center"],
            }
        )

    invoices_df = pl.DataFrame(invoices, schema=_INVOICE_SCHEMA)
    line_items_df = pl.DataFrame(line_items, schema=_LINE_ITEM_SCHEMA)
    return invoices_df, line_items_df


def _unique_invoice_id(rng: random.Random, used: set[str]) -> str:
    """A 6-digit numeric string that never starts with 0.

    Stored as text (per schema) and kept zero-free at the front on purpose, so
    the Step 3b 'leading-zero' trick (e.g. 234581 -> 0234581) creates a genuine
    near-duplicate rather than colliding with an existing id.
    """
    while True:
        candidate = str(rng.randint(100_000, 999_999))
        if candidate not in used:
            used.add(candidate)
            return candidate


_ITEM_WORDS = [
    "Consulting hours", "Software license", "Hardware unit", "Support plan",
    "Shipping fee", "Maintenance", "Training session", "Cloud storage",
    "Office supplies", "Installation",
]


def _item_description(rng: random.Random) -> str:
    return rng.choice(_ITEM_WORDS)
