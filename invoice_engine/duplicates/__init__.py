"""Fuzzy / ML near-duplicate detection - the centerpiece.

Scores how likely two invoices are the same bill across invoice number, vendor
name, amount, and date, catching near-duplicates that exact-match ERP checks
miss. Approach decided and built in Step 6."""
