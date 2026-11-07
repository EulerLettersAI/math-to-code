"""Data-fidelity metrics adapted from ifcfill-experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


NUMERIC_HELLINGER_BINS = 10


def cramers_v_squared(first: pd.Series, second: pd.Series) -> float:
    """Return the categorical association computed by SynthEval 1.6.2."""

    contingency = np.asarray(pd.crosstab(first, second, rownames=None, colnames=None))
    chi_squared = chi2_contingency(contingency)[0]
    observations = contingency.sum()
    smallest_dimension = min(contingency.shape) - 1
    return float(chi_squared / (observations * smallest_dimension + 1e-16))


def eta_squared(categories: pd.Series, measurements: pd.Series) -> float:
    """Return the categorical-to-numerical association used by SynthEval."""

    category_codes, _ = pd.factorize(categories)
    values = np.asarray(measurements, dtype=float)
    number_of_categories = category_codes.max() + 1
    group_means = np.zeros(number_of_categories)
    group_sizes = np.zeros(number_of_categories)
    for group in range(number_of_categories):
        group_values = values[category_codes == group]
        group_sizes[group] = len(group_values)
        group_means[group] = np.average(group_values)
    overall_mean = np.sum(group_means * group_sizes) / np.sum(group_sizes)
    between_group = np.sum(group_sizes * (group_means - overall_mean) ** 2)
    total = np.sum((values - overall_mean) ** 2)
    return 0.0 if between_group == 0 else float(between_group / total)


def _pairwise_matrix(
    data: pd.DataFrame,
    row_columns: list[str] | tuple[str, ...],
    column_columns: list[str] | tuple[str, ...],
    association,
) -> pd.DataFrame:
    values = [
        association(data[row], data[column])
        for row in row_columns
        for column in column_columns
    ]
    return pd.DataFrame(
        np.asarray(values).reshape(len(row_columns), len(column_columns)),
        index=row_columns,
        columns=column_columns,
    )


def mixed_association_matrix(
    data: pd.DataFrame,
    numerical_columns: list[str] | tuple[str, ...],
    categorical_columns: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Return SynthEval 1.6.2's executable mixed association matrix."""

    numerical_numerical = data[list(numerical_columns)].corr()
    categorical_categorical = _pairwise_matrix(
        data, categorical_columns, categorical_columns, cramers_v_squared
    )
    categorical_numerical = _pairwise_matrix(
        data, categorical_columns, numerical_columns, eta_squared
    )
    top = pd.concat([categorical_categorical, categorical_numerical], axis=1)
    bottom = pd.concat([categorical_numerical.T, numerical_numerical], axis=1)
    matrix = pd.concat([top, bottom], axis=0)
    return matrix + np.diag(1 - np.diag(matrix))


def correlation_matrix_difference(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    numerical_columns: list[str] | tuple[str, ...],
    categorical_columns: list[str] | tuple[str, ...],
) -> float:
    """Return the Frobenius norm of two mixed association matrices' difference."""

    first_matrix = mixed_association_matrix(first, numerical_columns, categorical_columns)
    second_matrix = mixed_association_matrix(second, numerical_columns, categorical_columns)
    return float(np.linalg.norm(first_matrix - second_matrix, ord="fro"))


def hellinger_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return the Hellinger distance between aligned probability vectors."""

    distance = float(np.sqrt(0.5 * np.sum((np.sqrt(first) - np.sqrt(second)) ** 2)))
    if distance < 0.0 or distance > 1.0:
        raise ValueError(f"Hellinger distance is out of bounds: {distance}")
    return distance


def _counts_to_probabilities(
    first_counts: np.ndarray,
    second_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if first_counts.sum() == 0 or second_counts.sum() == 0:
        raise ValueError("Cannot compute probabilities from empty counts")
    return (
        first_counts.astype(float) / first_counts.sum(),
        second_counts.astype(float) / second_counts.sum(),
    )


def categorical_probability_vectors(
    first: pd.Series,
    second: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the shared levels and two aligned empirical probability vectors."""

    first_values = first.to_numpy(copy=False)
    second_values = second.to_numpy(copy=False)
    levels = np.union1d(first_values, second_values)
    first_counts = np.bincount(np.searchsorted(levels, first_values), minlength=len(levels))
    second_counts = np.bincount(np.searchsorted(levels, second_values), minlength=len(levels))
    first_probabilities, second_probabilities = _counts_to_probabilities(
        first_counts, second_counts
    )
    return levels, first_probabilities, second_probabilities


def numeric_probability_vectors(
    first: pd.Series,
    second: pd.Series,
    *,
    bins: int = NUMERIC_HELLINGER_BINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return shared bin edges and two aligned histogram probability vectors."""

    first_values = pd.to_numeric(first, errors="raise").to_numpy(dtype=float)
    second_values = pd.to_numeric(second, errors="raise").to_numpy(dtype=float)
    pooled = np.concatenate([first_values, second_values])
    if not np.isfinite(pooled).all():
        raise ValueError("Numeric Hellinger inputs must contain only finite values")
    if float(pooled.min()) == float(pooled.max()):
        edges = np.array([pooled.min(), pooled.max()])
        return edges, np.array([1.0]), np.array([1.0])

    first_counts, edges = np.histogram(
        first_values,
        bins=bins,
        range=(float(pooled.min()), float(pooled.max())),
    )
    second_counts = np.histogram(second_values, bins=edges)[0]
    first_probabilities, second_probabilities = _counts_to_probabilities(
        first_counts, second_counts
    )
    return edges, first_probabilities, second_probabilities


def empirical_feature_hellinger_details(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    categorical_columns: list[str] | tuple[str, ...],
    numeric_columns: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Return one Hellinger distance for every requested feature."""

    if first.empty or second.empty:
        raise ValueError("Both DataFrames must contain at least one row")
    rows: list[dict[str, object]] = []
    for column in categorical_columns:
        _, first_p, second_p = categorical_probability_vectors(first[column], second[column])
        rows.append(
            {"column": column, "type": "categorical", "distance": hellinger_distance(first_p, second_p)}
        )
    for column in numeric_columns:
        _, first_p, second_p = numeric_probability_vectors(first[column], second[column])
        rows.append(
            {"column": column, "type": "numerical", "distance": hellinger_distance(first_p, second_p)}
        )
    if not rows:
        raise ValueError("At least one categorical or numeric column is required")
    return pd.DataFrame(rows)


def empirical_feature_hellinger_distance(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    categorical_columns: list[str] | tuple[str, ...],
    numeric_columns: list[str] | tuple[str, ...],
) -> float:
    """Return the average empirical per-feature Hellinger distance."""

    details = empirical_feature_hellinger_details(
        first,
        second,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
    )
    return float(details["distance"].mean())
