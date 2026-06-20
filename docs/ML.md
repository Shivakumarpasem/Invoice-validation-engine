# Machine learning layer

This project has two layers, and it's worth being precise about which is which.

## Today: a similarity / record-linkage engine

The duplicate detector compares invoice pairs and blends per-field similarity
scores into one confidence. The blend weights and the threshold are chosen by
analysis - the precision/recall sweep against ground truth - not learned by a
model. This is the technique family production entity-resolution systems use, and
it's the right transparent first layer.

## By design: the feature layer for a trained model

The project is built so a learned classifier drops straight on top, because the
hard part is already done:

- The four per-pair signals (`id_sim`, `name_sim`, `amount_sim`, `date_sim`) are
  an **ML feature vector** for each candidate pair.
- The ground-truth answer key (`duplicate_of`) is the **labels**.
- Blocking + scoring already turn raw invoices into a clean `features + label`
  table - the part that usually takes most of the work.

## The trained model

`python -m invoice_engine.ml.run` trains a **logistic-regression** classifier on
`features -> label` with a 70/30 train/test split, then evaluates it on the
held-out test set (pairs it never saw during training). Logistic regression is
chosen for explainability: every learned weight is readable, which matters in a
finance/audit setting.

What it shows:

- **Held-out performance** (precision / recall / F1 on unseen pairs) - the proof
  the model generalizes rather than memorizes.
- **The learned weights** - and notably, the model decides for itself which
  signals matter most, which can differ from the hand-tuned guess. For example it
  learns that exact amount match is the strongest signal, and that name
  similarity barely helps *inside a same-vendor block* (since nearly every pair
  there already shares the vendor name).

## The honest framing

The synthetic data separates cleanly, so the scores are high - that proves the
*mechanism and the evaluation are correct*, not that the numbers would be perfect
on messy real data. The point of this layer is the method (features, labels,
held-out evaluation, readable weights), and that a learned model reaches the same
conclusion as the hand-tuned rule from the data alone.

The core engine does **not** depend on this layer - it runs fully without
scikit-learn. The ML step is an optional add-on on top of the same features.
