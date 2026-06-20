"""Test the most important ingestion guarantee: leading zeros survive.

If a CSV's invoice_id "0412809" is read as the integer 412809, the leading-zero
forgery - the whole point of the project - is silently destroyed. This proves the
type-forcing read preserves it as text.
"""

import polars as pl

from invoice_engine.ingestion.load import _INVOICE_OVERRIDES


def test_invoice_id_is_forced_to_string():
    # The override must declare invoice_id as String, not let polars infer int.
    assert _INVOICE_OVERRIDES["invoice_id"] == pl.String


def test_leading_zero_survives_typed_read(tmp_path):
    # Write a tiny CSV with a leading-zero id, read it with the real overrides,
    # and confirm the zero is still there.
    csv = tmp_path / "invoices.csv"
    csv.write_text("record_id,invoice_id\n1,0412809\n2,412809\n")

    df = pl.read_csv(csv, schema_overrides=_INVOICE_OVERRIDES)
    ids = df.get_column("invoice_id").to_list()
    assert ids == ["0412809", "412809"]
    # And the two are distinct - the forgery is preserved as a real difference.
    assert ids[0] != ids[1]


def test_money_override_is_decimal():
    # Money must be exact Decimal, never float (avoids rounding in the sum check).
    assert isinstance(_INVOICE_OVERRIDES["invoice_amount"], pl.Decimal)
