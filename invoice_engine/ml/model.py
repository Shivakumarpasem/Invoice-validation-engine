"""Train and evaluate a logistic-regression duplicate classifier.

This is what makes the project genuinely ML: instead of US choosing the blend
weights and threshold, the model LEARNS them from labeled examples, and we judge
it on data it never saw during training (a held-out test set) - the standard way
ML models are evaluated, and the proof that it generalizes rather than memorizes.

Why logistic regression: it learns one weight per feature plus a bias, so the
result is fully readable ("the model learned amount matters most"). That
explainability is exactly what finance/audit settings need, and it lines up
directly against our hand-tuned weights for a clean "learned vs guessed"
comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from .dataset import FEATURE_COLUMNS

# Fixed seed so the train/test split (and thus the reported numbers) reproduce.
RANDOM_STATE = 42
TEST_FRACTION = 0.30


@dataclass
class TrainResult:
    model: LogisticRegression
    precision: float
    recall: float
    f1: float
    n_train: int
    n_test: int
    test_positives: int
    learned_weights: dict[str, float]
    intercept: float


def train_and_evaluate(X: np.ndarray, y: np.ndarray) -> TrainResult:
    """Split, train logistic regression, evaluate on the held-out test set.

    ``stratify=y`` keeps the same duplicate/non-duplicate ratio in both splits -
    important because duplicates are a small minority of all candidate pairs.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_FRACTION,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # class_weight="balanced": duplicates are rare among candidate pairs, so we
    # tell the model not to ignore the minority class just because it's small.
    model = LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )

    learned = dict(zip(FEATURE_COLUMNS, model.coef_[0].tolist()))
    return TrainResult(
        model=model,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        n_train=len(y_train),
        n_test=len(y_test),
        test_positives=int(y_test.sum()),
        learned_weights=learned,
        intercept=float(model.intercept_[0]),
    )
