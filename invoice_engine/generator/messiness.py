"""The messiness layer — deliberately plant the problems we want to catch.

Takes the clean invoices and:
  * DUPLICATE family — ADDS sneaky near-copies of real invoices (exact resubmit,
    leading-zero number, reworded vendor, tweaked number).
  * INTEGRITY family — CORRUPTS one field on existing clean invoices (broken sum,
    missing PO, bad date).

Everything planted is recorded in a **ground-truth** table (the answer key):
which record is a problem, what kind, and (for duplicates) which original it
copies. That answer key is what lets us measure the detector's precision/recall
later. Without it, we'd have no way to prove the engine actually works.
"""

from __future__ import annotations

import random
from datetime import timedelta
from decimal import Decimal

import polars as pl

from .config import GeneratorConfig
from .core import _INVOICE_SCHEMA, _LINE_ITEM_SCHEMA

_GROUND_TRUTH_SCHEMA = {
    "record_id": pl.Int64,  # the problem record
    "problem_family": pl.String,  # "duplicate" or "integrity"
    "problem_type": pl.String,  # e.g. "leading_zero"
    "duplicate_of": pl.Int64,  # original record_id (null for integrity errors)
    "detail": pl.String,  # human-readable note
}


def inject_messiness(
    invoices_df: pl.DataFrame, line_items_df: pl.DataFrame, config: GeneratorConfig
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return ``(invoices_df, line_items_df, ground_truth_df)`` with problems planted."""
    messer = _Messer(invoices_df, line_items_df, config)
    messer.plant_all()
    return messer.to_frames()


class _Messer:
    """Holds the growing pile of invoices and stamps problems onto it.

    Working with plain Python dicts (not polars) here is deliberate: editing
    individual rows and copying records is far simpler on dicts. We convert back
    to polars tables only at the very end.
    """

    def __init__(self, invoices_df, line_items_df, config: GeneratorConfig):
        self.config = config
        self.rng = random.Random(config.seed + 2)  # own stream, distinct from core

        self.invoices: list[dict] = invoices_df.to_dicts()
        self._by_id = {r["record_id"]: r for r in self.invoices}

        # line items grouped by their owning record_id, so we can copy an
        # invoice's lines when we duplicate it
        self.lines_by_record: dict[int, list[dict]] = {}
        for ln in line_items_df.to_dicts():
            self.lines_by_record.setdefault(ln["record_id"], []).append(ln)

        # next free surrogate key (duplicates we add get new unique record_ids)
        self.next_id = max(self._by_id) + 1

        # a shuffled bag of untouched clean invoices, so each problem targets a
        # different invoice and the answer key stays unambiguous
        self._untouched = list(self._by_id)
        self.rng.shuffle(self._untouched)

        self.ground_truth: list[dict] = []

    # ----- orchestration -----
    def plant_all(self):
        self.plant_exact_duplicates()
        self.plant_leading_zero()
        self.plant_reworded_vendor()
        self.plant_tweaked_number()
        self.plant_broken_sum()
        self.plant_missing_po()
        self.plant_bad_date()

    # ----- DUPLICATE family: add a sneaky copy of a real invoice -----
    def plant_exact_duplicates(self):
        for _ in range(self.config.n_exact_duplicate):
            original = self._take_original()
            self._add_duplicate(
                original, "exact_duplicate",
                "identical resubmission (same invoice number)",
                mutate=lambda inv: None,  # change nothing — a true exact copy
            )

    def plant_leading_zero(self):
        for _ in range(self.config.n_leading_zero):
            original = self._take_original()
            detail = f"'0' prefixed to number {original['invoice_id']}"
            self._add_duplicate(
                original, "leading_zero", detail,
                mutate=lambda inv: inv.__setitem__("invoice_id", "0" + inv["invoice_id"]),
            )

    def plant_reworded_vendor(self):
        for _ in range(self.config.n_reworded_vendor):
            original = self._take_original()
            new_name = _reword(original["party_name"], self.rng)
            new_number = _tweak_number(original["invoice_id"], self.rng)
            detail = f"vendor '{original['party_name']}' -> '{new_name}', number changed"

            def mutate(inv, new_name=new_name, new_number=new_number):
                inv["party_name"] = new_name
                inv["invoice_id"] = new_number

            self._add_duplicate(original, "reworded_vendor", detail, mutate)

    def plant_tweaked_number(self):
        for _ in range(self.config.n_tweaked_number):
            original = self._take_original()
            new_number = _tweak_number(original["invoice_id"], self.rng)
            detail = "same amount+date, different invoice number"

            def mutate(inv, new_number=new_number):
                inv["invoice_id"] = new_number

            self._add_duplicate(original, "tweaked_number", detail, mutate)

    # ----- INTEGRITY family: corrupt one field on an existing invoice -----
    def plant_broken_sum(self):
        for _ in range(self.config.n_broken_sum):
            inv = self._take_original()
            bump = Decimal(self.rng.choice([-150, -75, -50, 25, 80, 200]))
            old = inv["invoice_amount"]
            inv["invoice_amount"] = old + bump
            self._record_integrity(
                inv, "broken_sum", f"header {old} != sum of line items"
            )

    def plant_missing_po(self):
        planted = 0
        while planted < self.config.n_missing_po:
            inv = self._take_original()
            if inv["po_number"] is None:
                continue  # this one never had a PO; consume it and try another
            old = inv["po_number"]
            inv["po_number"] = None
            self._record_integrity(inv, "missing_po", f"PO {old} removed")
            planted += 1

    def plant_bad_date(self):
        for _ in range(self.config.n_bad_date):
            inv = self._take_original()
            inv["due_date"] = inv["invoice_date"] - timedelta(days=self.rng.randint(5, 30))
            self._record_integrity(inv, "bad_date", "due_date before invoice_date")

    # ----- shared helpers -----
    def _take_original(self) -> dict:
        """Pull one untouched clean invoice from the bag."""
        return self._by_id[self._untouched.pop()]

    def _add_duplicate(self, original, problem_type, detail, mutate):
        """Append a near-copy of ``original`` under a fresh record_id."""
        new_id = self.next_id
        self.next_id += 1

        copy_inv = dict(original)  # copy the header fields
        copy_inv["record_id"] = new_id
        mutate(copy_inv)  # apply the specific trick
        self.invoices.append(copy_inv)

        # copy the original's line items under the new record_id
        self.lines_by_record[new_id] = [
            {**ln, "record_id": new_id} for ln in self.lines_by_record[original["record_id"]]
        ]

        self.ground_truth.append(
            {
                "record_id": new_id,
                "problem_family": "duplicate",
                "problem_type": problem_type,
                "duplicate_of": original["record_id"],
                "detail": detail,
            }
        )

    def _record_integrity(self, inv, problem_type, detail):
        self.ground_truth.append(
            {
                "record_id": inv["record_id"],
                "problem_family": "integrity",
                "problem_type": problem_type,
                "duplicate_of": None,
                "detail": detail,
            }
        )

    def to_frames(self):
        all_lines = [ln for lines in self.lines_by_record.values() for ln in lines]
        invoices_df = pl.DataFrame(self.invoices, schema=_INVOICE_SCHEMA).sort("record_id")
        line_items_df = pl.DataFrame(all_lines, schema=_LINE_ITEM_SCHEMA).sort("record_id")
        ground_truth_df = pl.DataFrame(
            self.ground_truth, schema=_GROUND_TRUTH_SCHEMA
        ).sort("record_id")
        return invoices_df, line_items_df, ground_truth_df


# --------- text tricks (used to forge near-duplicates) ---------
def _reword(name: str, rng: random.Random) -> str:
    """Reword a company name the way a real resubmission might differ."""
    transforms = [
        lambda s: s.upper(),
        lambda s: s.replace(" and ", " & "),
        lambda s: s.replace(",", ""),
        lambda s: s + " Inc",
        lambda s: s.replace("Ltd", "Limited"),
    ]
    new = name
    for t in rng.sample(transforms, k=rng.randint(1, 2)):
        new = t(new)
    return new if new != name else name + " Ltd"


def _tweak_number(num: str, rng: random.Random) -> str:
    """Change the printed number slightly so an exact-match check fails."""
    if num.isdigit() and rng.random() < 0.5:
        i = rng.randrange(len(num))
        new = num[:i] + str(rng.randint(0, 9)) + num[i + 1 :]
        return new if new != num else num + "1"
    return num + "-A"
