"""Reusable example tables and views for the dataset-comparison lessons."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


NUMERIC_COLUMNS = ("age", "income_k")
CATEGORICAL_COLUMN = "region"
MISSING_LEVEL = "__MISSING__"
METRIC_NUMERIC_COLUMNS = ("age", "income_k")
METRIC_CATEGORICAL_COLUMNS = (
    "region__code",
    "age__missing",
    "income_k__missing",
    "outcome",
)
TARGET_COLUMN = "outcome"


@dataclass(frozen=True)
class ExampleViews:
    """Fresh copies of every shared object used across the lesson series."""

    X: pd.DataFrame
    Y: pd.DataFrame
    U: pd.DataFrame
    V: pd.DataFrame
    M_X: pd.DataFrame
    M_Y: pd.DataFrame
    M_U: pd.DataFrame
    M_V: pd.DataFrame
    X_tilde: pd.DataFrame
    Y_tilde: pd.DataFrame
    U_tilde: pd.DataFrame
    V_tilde: pd.DataFrame
    numeric_parameters: pd.DataFrame
    region_to_code: dict[str, int]
    region_mapping: pd.DataFrame


@dataclass(frozen=True)
class DependenceExample:
    """Larger raw tables needed by the classifier-based dependence metric."""

    X: pd.DataFrame
    Y: pd.DataFrame
    V: pd.DataFrame


def make_example_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the same small, mixed-type X and Y tables for every lesson."""

    X = pd.DataFrame(
        {
            "age": pd.Series([24, 31, pd.NA, 46, 58, 39], dtype="Int64"),
            "income_k": pd.Series([32, 48, 55, pd.NA, 91, 67], dtype="Float64"),
            "region": pd.Series(
                ["North", "South", "North", "West", pd.NA, "South"],
                dtype="string",
            ),
            "outcome": pd.Series([0, 0, 1, 1, 1, 0], dtype="Int64"),
        }
    )
    X.index = [f"x{i}" for i in range(1, len(X) + 1)]

    Y = pd.DataFrame(
        {
            "age": pd.Series([25, pd.NA, 44, 60, 37], dtype="Int64"),
            "income_k": pd.Series([35, 51, pd.NA, 88, 64], dtype="Float64"),
            "region": pd.Series(
                ["North", "South", "West", pd.NA, "East"],
                dtype="string",
            ),
            "outcome": pd.Series([0, 1, 1, 1, 0], dtype="Int64"),
        }
    )
    Y.index = [f"y{i}" for i in range(1, len(Y) + 1)]
    return X, Y


def make_holdout_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return independent utility and missingness holdouts with the same schema."""

    U = pd.DataFrame(
        {
            "age": pd.Series([27, 34, 42, 49, pd.NA, 55, 30, 63, 45, 38], dtype="Int64"),
            "income_k": pd.Series(
                [36, 49, 60, pd.NA, 82, 85, 44, 96, 70, 58], dtype="Float64"
            ),
            "region": pd.Series(
                ["North", "South", "West", "North", "East", "South", pd.NA, "West", "North", "East"],
                dtype="string",
            ),
            "outcome": pd.Series([0, 0, 1, 1, 1, 1, 0, 1, 1, 0], dtype="Int64"),
        }
    )
    U.index = [f"u{i}" for i in range(1, len(U) + 1)]

    V = pd.DataFrame(
        {
            "age": pd.Series([26, 33, 41, pd.NA, 52, 59, 36, 47, 62, 29], dtype="Int64"),
            "income_k": pd.Series(
                [34, pd.NA, 57, 63, 78, 90, 50, 72, 94, 40], dtype="Float64"
            ),
            "region": pd.Series(
                ["North", "South", "West", "East", pd.NA, "South", "North", "West", "East", "North"],
                dtype="string",
            ),
            "outcome": pd.Series([0, 0, 1, 1, 1, 1, 0, 1, 1, 0], dtype="Int64"),
        }
    )
    V.index = [f"v{i}" for i in range(1, len(V) + 1)]
    return U, V


def learn_shared_transformation(
    X: pd.DataFrame,
    Y: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int], pd.DataFrame]:
    """Learn numerical parameters and category codes from the pooled pair."""

    if not X.columns.equals(Y.columns):
        raise ValueError("X and Y must contain the same columns in the same order")
    if not X.dtypes.astype(str).equals(Y.dtypes.astype(str)):
        raise ValueError("X and Y must use the same datatype for every column")

    pooled = pd.concat([X, Y], axis=0)
    numeric_parameters = pd.DataFrame(
        {
            "median": pooled[list(NUMERIC_COLUMNS)].median(),
            "minimum": pooled[list(NUMERIC_COLUMNS)].min(),
            "maximum": pooled[list(NUMERIC_COLUMNS)].max(),
        }
    )
    numeric_parameters["range"] = (
        numeric_parameters["maximum"] - numeric_parameters["minimum"]
    )

    observed_regions = sorted(pooled[CATEGORICAL_COLUMN].dropna().unique())
    region_levels = [*observed_regions, MISSING_LEVEL]
    region_to_code = {level: code for code, level in enumerate(region_levels)}
    region_mapping = pd.DataFrame(
        {"region level": region_to_code.keys(), "shared code": region_to_code.values()}
    )
    return numeric_parameters, region_to_code, region_mapping


def transform_to_metric_view(
    frame: pd.DataFrame,
    numeric_parameters: pd.DataFrame,
    region_to_code: dict[str, int],
) -> pd.DataFrame:
    """Apply the fitted shared rules to one table without changing its rows."""

    transformed = pd.DataFrame(index=frame.index)
    for column in NUMERIC_COLUMNS:
        parameters = numeric_parameters.loc[column]
        missing = frame[column].isna()
        filled = frame[column].astype("Float64").fillna(parameters["median"])
        transformed[column] = (
            (filled.astype(float) - parameters["minimum"]) / parameters["range"]
        )
        transformed[f"{column}__missing"] = missing.astype("int8")

    categorical = frame[CATEGORICAL_COLUMN].fillna(MISSING_LEVEL)
    transformed[f"{CATEGORICAL_COLUMN}__code"] = (
        categorical.map(region_to_code).astype("int64")
    )
    transformed["outcome"] = frame["outcome"].astype("int64")
    return transformed


def make_example_views() -> ExampleViews:
    """Construct every shared object used by the subsequent notebooks."""

    X, Y = make_example_tables()
    U, V = make_holdout_tables()
    M_X = X.isna().astype("int8")
    M_Y = Y.isna().astype("int8")
    M_U = U.isna().astype("int8")
    M_V = V.isna().astype("int8")
    numeric_parameters, region_to_code, region_mapping = learn_shared_transformation(X, Y)
    X_tilde = transform_to_metric_view(X, numeric_parameters, region_to_code)
    Y_tilde = transform_to_metric_view(Y, numeric_parameters, region_to_code)
    U_tilde = transform_to_metric_view(U, numeric_parameters, region_to_code)
    V_tilde = transform_to_metric_view(V, numeric_parameters, region_to_code)
    return ExampleViews(
        X=X,
        Y=Y,
        U=U,
        V=V,
        M_X=M_X,
        M_Y=M_Y,
        M_U=M_U,
        M_V=M_V,
        X_tilde=X_tilde,
        Y_tilde=Y_tilde,
        U_tilde=U_tilde,
        V_tilde=V_tilde,
        numeric_parameters=numeric_parameters,
        region_to_code=region_to_code,
        region_mapping=region_mapping,
    )


def _dependence_frame(n_rows: int, seed: int, *, dependent_missingness: bool) -> pd.DataFrame:
    """Generate one table with controlled missingness in ``income_k``."""

    import numpy as np

    rng = np.random.default_rng(seed)
    age = rng.normal(45.0, 12.0, size=n_rows)
    income = 25.0 + 0.9 * age + rng.normal(0.0, 8.0, size=n_rows)
    region = rng.choice(["North", "South", "East", "West"], size=n_rows)
    outcome = (income + rng.normal(0.0, 8.0, size=n_rows) > 67.0).astype(int)

    if dependent_missingness:
        score = (age - 45.0) / 12.0 + 0.4 * (region == "South") + rng.normal(
            0.0, 0.55, size=n_rows
        )
        missing = score > 0.35
    else:
        missing = rng.random(n_rows) < 0.38
    income[missing] = np.nan

    return pd.DataFrame(
        {
            "age": pd.Series(age, dtype="Float64"),
            "income_k": pd.Series(income, dtype="Float64"),
            "region": pd.Series(region, dtype="string"),
            "outcome": pd.Series(outcome, dtype="Int64"),
        }
    )


def make_dependence_example(n_rows: int = 180) -> DependenceExample:
    """Return reference, comparison, and evaluation tables for dependence."""

    return DependenceExample(
        X=_dependence_frame(n_rows, seed=101, dependent_missingness=True),
        Y=_dependence_frame(n_rows, seed=202, dependent_missingness=False),
        V=_dependence_frame(n_rows, seed=303, dependent_missingness=True),
    )
