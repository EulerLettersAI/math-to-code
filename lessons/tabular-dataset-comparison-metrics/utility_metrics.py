"""Utility metrics adapted from ifcfill-experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC


def _models() -> tuple[tuple[str, object], ...]:
    return (
        ("Random forest", RandomForestClassifier(n_estimators=10, max_depth=15, random_state=42)),
        ("AdaBoost", AdaBoostClassifier(n_estimators=10, learning_rate=1, random_state=42)),
        ("RBF SVC", SVC(kernel="rbf", random_state=42)),
        ("Logistic regression", LogisticRegression(solver="saga", max_iter=5000, random_state=42)),
    )


def holdout_auroc_details(
    *,
    reference_data: pd.DataFrame,
    comparison_data: pd.DataFrame,
    holdout_data: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    """Return one holdout AUROC per model and training table."""

    holdout_x = holdout_data.drop(columns=[target_column])
    holdout_y = holdout_data[target_column]
    rows = []
    for model_name, model in _models():
        row: dict[str, object] = {"model": model_name}
        for label, training_data in (("AUROC_X", reference_data), ("AUROC_Y", comparison_data)):
            train_x = training_data.drop(columns=[target_column])
            train_y = training_data[target_column]
            fitted_model = clone(model) if train_y.nunique() >= 2 else DummyClassifier(
                strategy="most_frequent"
            )
            fitted_model.fit(train_x, train_y)
            scores = _positive_class_scores(fitted_model, holdout_x)
            row[label] = float(roc_auc_score(holdout_y, scores))
        rows.append(row)
    details = pd.DataFrame(rows)
    details["difference_Y_minus_X"] = details["AUROC_Y"] - details["AUROC_X"]
    return details


def average_holdout_auroc(
    *,
    reference_data: pd.DataFrame,
    comparison_data: pd.DataFrame,
    holdout_data: pd.DataFrame,
    target_column: str,
) -> tuple[float, float]:
    """Return average AUROC for models trained on each side and tested on one holdout."""

    details = holdout_auroc_details(
        reference_data=reference_data,
        comparison_data=comparison_data,
        holdout_data=holdout_data,
        target_column=target_column,
    )
    return float(details["AUROC_X"].mean()), float(details["AUROC_Y"].mean())


def independent_holdout_auroc_difference(
    *,
    reference_data: pd.DataFrame,
    comparison_data: pd.DataFrame,
    holdout_data: pd.DataFrame,
    target_column: str,
) -> float:
    """Return average holdout AUROC(Y-trained) minus AUROC(X-trained)."""

    average_x, average_y = average_holdout_auroc(
        reference_data=reference_data,
        comparison_data=comparison_data,
        holdout_data=holdout_data,
        target_column=target_column,
    )
    return average_y - average_x


def _positive_class_scores(model, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        if probabilities.shape[1] == 1:
            return np.zeros(len(features))
        return probabilities[:, 1]
    return model.decision_function(features)
