"""Missingness-fidelity metrics adapted from ifcfill-experiments."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from data_fidelity_metrics import hellinger_distance


MIN_DEPENDENCE_CASES = 5


def marginal_missingness_details(first: pd.DataFrame, second: pd.DataFrame) -> pd.DataFrame:
    """Return missingness probabilities and Hellinger distance by active column."""

    _validate_pair(first, second)
    second = second[first.columns]
    rows = []
    for column in first.columns:
        if not (first[column].isna().any() or second[column].isna().any()):
            continue
        first_missing = float(first[column].isna().mean())
        second_missing = float(second[column].isna().mean())
        first_p = np.array([1.0 - first_missing, first_missing])
        second_p = np.array([1.0 - second_missing, second_missing])
        rows.append(
            {
                "column": column,
                "P_X(missing)": first_missing,
                "P_Y(missing)": second_missing,
                "distance": hellinger_distance(first_p, second_p),
            }
        )
    return pd.DataFrame(rows, columns=["column", "P_X(missing)", "P_Y(missing)", "distance"])


def joint_missingness_details(first: pd.DataFrame, second: pd.DataFrame) -> pd.DataFrame:
    """Return aligned empirical probabilities of row-wise missingness patterns."""

    _validate_pair(first, second)
    second = second[first.columns]
    first_patterns = first.isna().astype(np.uint8).apply(tuple, axis=1)
    second_patterns = second.isna().astype(np.uint8).apply(tuple, axis=1)
    first_p = first_patterns.value_counts(normalize=True)
    second_p = second_patterns.value_counts(normalize=True)
    patterns = first_p.index.union(second_p.index)
    return pd.DataFrame(
        {
            "pattern": list(patterns),
            "P_X(pattern)": first_p.reindex(patterns, fill_value=0).to_numpy(dtype=float),
            "P_Y(pattern)": second_p.reindex(patterns, fill_value=0).to_numpy(dtype=float),
        }
    )


def missingness_hellinger_distance(
    first: pd.DataFrame,
    second: pd.DataFrame,
    mode: Literal["marginal", "joint"],
) -> float:
    """Compare marginal or joint empirical missingness distributions."""

    if mode == "marginal":
        details = marginal_missingness_details(first, second)
        return float(details["distance"].mean()) if not details.empty else 0.0
    if mode == "joint":
        details = joint_missingness_details(first, second)
        return hellinger_distance(
            details["P_X(pattern)"].to_numpy(), details["P_Y(pattern)"].to_numpy()
        )
    raise ValueError("mode must be either 'marginal' or 'joint'")


def missingness_dependence_details(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    categorical_columns: list[str] | tuple[str, ...],
    datetime_columns: list[str] | tuple[str, ...] = (),
    random_state: int,
) -> pd.DataFrame:
    """Return holdout AUCs and their gap for every eligible missing column."""

    if not set(reference.columns) == set(comparison.columns) == set(evaluation.columns):
        raise ValueError("All DataFrames must contain the same column names")
    columns = list(reference.columns)
    reference, comparison, evaluation = (
        _with_datetime_days(frame[columns], datetime_columns)
        for frame in (reference, comparison, evaluation)
    )

    rows = []
    for column in columns:
        reference_missing = reference[column].isna().to_numpy(dtype=int)
        evaluation_missing = evaluation[column].isna().to_numpy(dtype=int)
        if min(_minority_count(reference_missing), _minority_count(evaluation_missing)) < (
            MIN_DEPENDENCE_CASES
        ):
            continue

        features = [other for other in columns if other != column]
        aucs = []
        for training, training_missing in (
            (reference, reference_missing),
            (comparison, comparison[column].isna().to_numpy(dtype=int)),
        ):
            if training_missing.min() == training_missing.max():
                aucs.append(0.5)
                continue
            model = _missingness_model(features, categorical_columns, random_state)
            model.fit(training[features], training_missing)
            scores = model.predict_proba(evaluation[features])[:, 1]
            aucs.append(float(roc_auc_score(evaluation_missing, scores)))
        rows.append(
            {"column": column, "AUC_X": aucs[0], "AUC_Y": aucs[1], "gap": aucs[0] - aucs[1]}
        )
    return pd.DataFrame(rows, columns=["column", "AUC_X", "AUC_Y", "gap"])


def missingness_dependence_gap(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    categorical_columns: list[str] | tuple[str, ...],
    datetime_columns: list[str] | tuple[str, ...] = (),
    random_state: int,
) -> float:
    """Return mean holdout AUC(reference) minus AUC(comparison)."""

    details = missingness_dependence_details(
        reference,
        comparison,
        evaluation,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
        random_state=random_state,
    )
    return float(details["gap"].mean()) if not details.empty else float("nan")


def _validate_pair(first: pd.DataFrame, second: pd.DataFrame) -> None:
    if set(first.columns) != set(second.columns):
        raise ValueError("Both DataFrames must contain the same column names")
    if first.empty or second.empty:
        raise ValueError("Both DataFrames must contain at least one row")


def _minority_count(indicator: np.ndarray) -> int:
    return int(min(indicator.sum(), len(indicator) - indicator.sum()))


def _with_datetime_days(
    frame: pd.DataFrame,
    datetime_columns: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    converted = frame.copy()
    for column in datetime_columns:
        numeric = pd.to_numeric(converted[column], errors="coerce")
        if numeric.notna().sum() == converted[column].notna().sum():
            converted[column] = numeric.astype(float)
        else:
            dates = pd.to_datetime(converted[column], errors="coerce")
            converted[column] = (
                (dates - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1)
            ).astype(float)
    return converted


def _missingness_model(
    features: list[str],
    categorical_columns: list[str] | tuple[str, ...],
    random_state: int,
):
    categorical = [column for column in features if column in categorical_columns]
    numeric = [column for column in features if column not in categorical_columns]
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(
                    FunctionTransformer(_as_float),
                    SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                ),
                numeric,
            ),
            (
                "categorical",
                make_pipeline(
                    FunctionTransformer(_as_missing_aware_text),
                    OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False),
                ),
                categorical,
            ),
        ]
    )
    return make_pipeline(preprocessor, HistGradientBoostingClassifier(random_state=random_state))


def _as_float(frame: pd.DataFrame) -> np.ndarray:
    return frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def _as_missing_aware_text(frame: pd.DataFrame) -> np.ndarray:
    return frame.astype(object).where(frame.notna(), "__missing__").astype(str).to_numpy()
