# Running the engine

Full setup and run guide. For a one-look quickstart, see the main
[README](../README.md).

## Setup

Requires Python 3.11+ and the packages in `requirements.txt`.

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# macOS/Linux:
source .venv/bin/activate && pip install -r requirements.txt
```

> **Windows:** prefix every command below with the venv interpreter, e.g.
> `.\.venv\Scripts\python.exe -m invoice_engine.reporting.run`

## Core pipeline (no ML required)

The full duplicate-detection product runs without any machine-learning
dependency. Run these **in order** - each step feeds the next, and the report at
the end reads what the earlier steps wrote to SQLite:

```bash
python -m invoice_engine.generator.generate    # 1. build the synthetic dataset (CSV)
python -m invoice_engine.ingestion.build        # 2. load + clean into SQLite
python -m invoice_engine.validation.run         # 3. integrity checks  -> validation_findings
python -m invoice_engine.scoring.run            # 4. detect duplicates + build the review queue
python -m invoice_engine.reporting.run          # 5. print the CLI report
```

## Seeing the report again

Once the pipeline has run, you only need the last command:

```bash
python -m invoice_engine.reporting.run                      # the sectioned report
python -m invoice_engine.reporting.run --top 20             # show 20 worst cases
python -m invoice_engine.reporting.run --all                # dump the full worklist
python -m invoice_engine.reporting.run --csv review.csv     # export the worklist to CSV
```

## The detector's proof, on its own

The side-by-side vs naive exact-match, with the full precision/recall sweep across
confidence thresholds:

```bash
python -m invoice_engine.duplicates.run
```

## Optional: train the ML model

This is a separate, optional layer - the core pipeline above does not need it. It
trains a classifier on the similarity features the detector produces and reports
how it performs on a held-out test set:

```bash
python -m invoice_engine.ml.run
```

See [ML layer](ML.md) for what it does and why.

## Using your own data

To run on real ERP invoices instead of the synthetic set, see
[Using your own ERP data](USE_YOUR_OWN_DATA.md). You skip step 1 and run steps
2 through 5 exactly as above.

## Rebuild-order note

`ingestion.build` rebuilds the database from scratch, which clears the findings
tables. So after re-running it, re-run `validation.run` and `scoring.run` before
`reporting.run`. The order above always works from a clean slate.
