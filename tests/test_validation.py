"""Tests for the integrity validation checks.

These prove the checks actually catch the problems they target - a broken-sum or
bad-date invoice gets flagged, and a clean one does not.
"""

from datetime import date
from decimal import Decimal

import polars as pl

from invoice_engine.validation.checks import check_date_sanity, check_line_item_sum


def _invoices(rows):
    return pl.DataFrame(
        rows,
        schema={
            "record_id": pl.Int64,
            "invoice_amount": pl.Decimal(scale=2),
            "invoice_date": pl.Date,
            "due_date": pl.Date,
        },
        orient="row",
    )


def _lines(rows):
    return pl.DataFrame(
        rows,
        schema={"record_id": pl.Int64, "line_total": pl.Decimal(scale=2)},
        orient="row",
    )


def test_line_item_sum_flags_mismatch():
    # Header says 100 but the single line totals 90 -> must be flagged.
    inv = _invoices([(1, Decimal("100.00"), date(2026, 1, 1), date(2026, 2, 1))])
    lines = _lines([(1, Decimal("90.00"))])
    findings = check_line_item_sum(inv, lines)
    assert findings.height == 1
    assert findings.row(0, named=True)["record_id"] == 1


def test_line_item_sum_passes_when_balanced():
    inv = _invoices([(1, Decimal("100.00"), date(2026, 1, 1), date(2026, 2, 1))])
    lines = _lines([(1, Decimal("60.00")), (1, Decimal("40.00"))])
    assert check_line_item_sum(inv, lines).height == 0


def test_date_sanity_flags_due_before_invoice():
    # due_date before invoice_date is impossible.
    inv = _invoices([(1, Decimal("100.00"), date(2026, 2, 1), date(2026, 1, 1))])
    findings = check_date_sanity(inv, _lines([(1, Decimal("100.00"))]))
    assert findings.height == 1


def test_date_sanity_passes_for_valid_dates():
    inv = _invoices([(1, Decimal("100.00"), date(2026, 1, 1), date(2026, 2, 1))])
    assert check_date_sanity(inv, _lines([(1, Decimal("100.00"))])).height == 0
