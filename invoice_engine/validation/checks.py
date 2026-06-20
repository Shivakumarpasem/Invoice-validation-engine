"""Validation checks (integrity + match readiness).

Each check returns a polars frame of FINDINGS in one uniform shape:
    record_id | check_name | severity | detail
so every check stacks into a single tidy findings table.

NOTE: these checks catch INTEGRITY problems. They do NOT catch duplicates - that
is deliberately the job of the fuzzy detector (Step 6). Validation answers "is
this one invoice internally valid?"; duplicate detection answers "is this the
same bill as another?".
"""

from __future__ import annotations

from decimal import Decimal

import polars as pl

FINDINGS_SCHEMA = {
    "record_id": pl.Int64,
    "check_name": pl.String,
    "severity": pl.String,  # "error" (clearly wrong) or "warning" (needs a human)
    "detail": pl.String,
}

_REQUIRED = ["invoice_id", "party_id", "invoice_date", "invoice_amount"]


def _finding(df: pl.DataFrame, check_name: str, severity: str, detail: pl.Expr) -> pl.DataFrame:
    """Shape any filtered frame into the uniform findings layout."""
    return df.select(
        pl.col("record_id"),
        pl.lit(check_name).alias("check_name"),
        pl.lit(severity).alias("severity"),
        detail.alias("detail"),
    )


def check_line_item_sum(invoices: pl.DataFrame, line_items: pl.DataFrame) -> pl.DataFrame:
    """Header total must equal the sum of its line items."""
    sums = line_items.group_by("record_id").agg(
        pl.col("line_total").sum().alias("line_sum")
    )
    joined = invoices.join(sums, on="record_id", how="left").with_columns(
        pl.col("line_sum").fill_null(Decimal("0.00"))
    )
    bad = joined.filter(pl.col("invoice_amount") != pl.col("line_sum"))
    return _finding(
        bad, "line_item_sum", "error",
        pl.format("header {} != sum of lines {}", "invoice_amount", "line_sum"),
    )


def check_required_fields(invoices: pl.DataFrame, line_items: pl.DataFrame) -> pl.DataFrame:
    """Core fields that must never be empty."""
    missing = pl.concat_str(
        [
            pl.when(pl.col(c).is_null()).then(pl.lit(c + " ")).otherwise(pl.lit(""))
            for c in _REQUIRED
        ]
    ).str.strip_chars()
    bad = invoices.filter(
        pl.any_horizontal(pl.col(c).is_null() for c in _REQUIRED)
    ).with_columns(missing.alias("_missing"))
    return _finding(
        bad, "required_fields", "error", pl.format("missing required: {}", "_missing")
    )


def check_date_sanity(invoices: pl.DataFrame, line_items: pl.DataFrame) -> pl.DataFrame:
    """A due date before the invoice date is impossible."""
    bad = invoices.filter(pl.col("due_date") < pl.col("invoice_date"))
    return _finding(
        bad, "date_sanity", "error",
        pl.format("due_date {} before invoice_date {}", "due_date", "invoice_date"),
    )


def check_missing_po(invoices: pl.DataFrame, line_items: pl.DataFrame) -> pl.DataFrame:
    """No PO -> can't be three-way matched -> flag for human review.

    A 'warning', not an 'error': some invoices legitimately have no PO. The data
    alone can't tell a removed PO from one that never existed, so we surface it
    rather than reject it. (A full three-way match of qty/price against PO +
    receipt data is deferred - we don't generate PO/receipt lines in v1.)
    """
    bad = invoices.filter(pl.col("po_number").is_null())
    return _finding(
        bad, "missing_po", "warning", pl.lit("no PO number; cannot three-way match")
    )


ALL_CHECKS = [
    check_line_item_sum,
    check_required_fields,
    check_date_sanity,
    check_missing_po,
]


def run_all_checks(invoices: pl.DataFrame, line_items: pl.DataFrame) -> pl.DataFrame:
    """Run every check and stack the findings into one table."""
    findings = [check(invoices, line_items) for check in ALL_CHECKS]
    return pl.concat(findings, how="vertical").cast(FINDINGS_SCHEMA)  # type: ignore[arg-type]
