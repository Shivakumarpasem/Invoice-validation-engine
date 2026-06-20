"""Configuration for the synthetic data generator.

One dataclass holds every dial, so the dataset's size and 'personality' are
controlled in a single place (and are easy to change for tests vs. demos).
Messiness counts are added in Step 3b; Step 3a only needs the base settings.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeneratorConfig:
    # --- Reproducibility ---
    seed: int = 42  # fixed seed => same dataset every run (vital for tests)

    # --- Volume ---
    n_invoices: int = 2_000  # modest default; crank up later for scale tests
    n_parties: int = 40  # fixed pool of vendors/customers that *recur*

    # --- AP / AR split ---
    ap_ratio: float = 0.6  # 60% AP (vendor bills), 40% AR (customer invoices)

    # --- Line items per invoice ---
    min_lines: int = 1
    max_lines: int = 6

    # --- Money ---
    currency: str = "USD"  # multi-currency is out of v1 scope
    tax_rate: str = "0.08"  # kept as str -> Decimal, to stay exact

    # --- Probabilities for optional fields on *clean* invoices ---
    p_has_po: float = 0.8  # most invoices reference a PO; some legitimately don't
    p_paid: float = 0.55  # share already paid (gets a payment_date)

    # --- Messiness counts (Step 3b): how many of each problem to plant ---
    # "Test-coverage" mix: known, fixed counts so we can grade the detector.
    # DUPLICATE family — each ADDS a sneaky near-copy of a real invoice:
    n_exact_duplicate: int = 40  # same printed number (exact-match WOULD catch this)
    n_leading_zero: int = 40  # "0" prefixed to the number (exact-match misses)
    n_reworded_vendor: int = 40  # vendor name reworded, number changed
    n_tweaked_number: int = 40  # same amount+date, different number
    # INTEGRITY family — each CORRUPTS one field on an existing clean invoice:
    n_broken_sum: int = 50  # header total no longer equals the line items
    n_missing_po: int = 40  # PO removed from an invoice that had one
    n_bad_date: int = 30  # due_date set before invoice_date
