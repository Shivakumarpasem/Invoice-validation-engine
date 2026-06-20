# Using your own ERP data

The engine does not care *where* the invoices come from - it only cares about
their **shape**. Everything after ingestion (cleaning, validation, duplicate
detection, the review queue, the report) reads from one fixed schema. So plugging
in real data (e.g. an export from Oracle / SAP AP tables) is essentially one job:
**get your invoices into that schema, then skip generation and run the rest.**

## Step A - export your invoices to two CSV files

Most ERP reporting teams can produce these from existing AP tables or a scheduled
extract.

`invoices.csv` - one row per invoice header:

| column | meaning |
|---|---|
| `record_id` | a unique row id (any integer; e.g. the ERP's internal key) |
| `invoice_id` | the printed invoice number - **keep leading zeros** |
| `direction` | `AP` (vendor bills) or `AR` (customer invoices) |
| `party_id` | the vendor/customer master id |
| `party_name` | the vendor/customer name |
| `invoice_date`, `due_date`, `payment_date` | dates (`YYYY-MM-DD`; payment may be blank) |
| `invoice_amount`, `tax_amount` | money (plain decimals, e.g. `1234.56`) |
| `currency`, `payment_terms`, `po_number`, `status`, `gl_account_code`, `cost_center` | as available (`po_number` may be blank) |

`line_items.csv` - one row per line:

| column | meaning |
|---|---|
| `record_id` | links to the invoice above |
| `description`, `qty`, `unit_price`, `line_total` | the line detail |

## Step B - point the loader at your files

Drop your two CSVs in `data/raw/` (replacing the generated ones), keeping the
column names above. The type-forcing loader (`invoice_engine/ingestion/load.py`)
already preserves leading zeros and exact decimals, so `00123` stays `00123`.

> Reading straight from a database instead of CSV? `load.py` is the single,
> isolated place to change - swap the `read_csv` calls for a SQL query against
> your AP tables (or your daily extract). Nothing downstream changes.

## Step C - run the pipeline (skip generation)

```bash
python -m invoice_engine.ingestion.build     # load YOUR csvs into SQLite
python -m invoice_engine.validation.run      # integrity checks
python -m invoice_engine.scoring.run         # detect duplicates + build the queue
python -m invoice_engine.reporting.run       # see the report on your data
```

## Note on real data

Real invoices have no built-in answer key, so the precision/recall grading (which
compares against planted problems) applies to the synthetic set only. For real
data you would confirm a sample of the flagged pairs once and adjust the
confidence threshold in `invoice_engine/duplicates/persist.py` to fit how strict
you want the queue.
