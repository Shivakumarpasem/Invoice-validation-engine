"""Machine-learning layer (Step 10).

Trains a classifier on the per-pair similarity features (built by the duplicate
detector) with the ground-truth answer key as labels - so the model LEARNS the
weights and decision boundary instead of us hand-setting them, and is evaluated
on a held-out test set.
"""
