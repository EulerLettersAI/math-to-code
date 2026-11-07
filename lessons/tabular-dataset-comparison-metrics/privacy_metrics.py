"""Record-level privacy metrics adapted from ifcfill-experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


EXACT_MATCH_DECIMALS = 9
DISTANCE_CHUNK_ROWS = 128


def exact_match_indicators(reference: pd.DataFrame, comparison: pd.DataFrame) -> pd.Series:
    """Indicate which comparison rows exactly match any reference row."""

    reference_keys = set(_row_hashes(reference))
    indicators = _row_hashes(comparison).isin(reference_keys)
    indicators.index = comparison.index
    return indicators


def exact_match_rate(reference: pd.DataFrame, comparison: pd.DataFrame) -> float:
    """Return the share of comparison rows identical to a reference row."""

    return float(exact_match_indicators(reference, comparison).mean())


def closest_record_distances(
    records: pd.DataFrame,
    released: pd.DataFrame,
    *,
    categorical_columns: list[str] | tuple[str, ...],
) -> np.ndarray:
    """Return each record's Gower-type distance to its closest released row."""

    categorical = [column for column in records.columns if column in categorical_columns]
    numeric = [column for column in records.columns if column not in categorical_columns]
    released_numeric = released[numeric].to_numpy(dtype=float)
    released_categorical = released[categorical].to_numpy()
    record_numeric = records[numeric].to_numpy(dtype=float)
    record_categorical = records[categorical].to_numpy()

    closest = np.empty(len(records))
    for start in range(0, len(records), DISTANCE_CHUNK_ROWS):
        rows = slice(start, start + DISTANCE_CHUNK_ROWS)
        total = np.abs(record_numeric[rows, None, :] - released_numeric[None, :, :]).sum(axis=2)
        total += (
            record_categorical[rows, None, :] != released_categorical[None, :, :]
        ).sum(axis=2)
        closest[rows] = total.min(axis=1) / len(records.columns)
    return closest


def distance_membership_details(
    members: pd.DataFrame,
    non_members: pd.DataFrame,
    released: pd.DataFrame,
    *,
    categorical_columns: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Return membership labels, closest distances, and attack scores."""

    member_distances = closest_record_distances(
        members, released, categorical_columns=categorical_columns
    )
    non_member_distances = closest_record_distances(
        non_members, released, categorical_columns=categorical_columns
    )
    return pd.DataFrame(
        {
            "group": ["member"] * len(member_distances)
            + ["non-member"] * len(non_member_distances),
            "label": np.r_[np.ones(len(member_distances)), np.zeros(len(non_member_distances))],
            "closest_distance": np.r_[member_distances, non_member_distances],
            "attack_score": -np.r_[member_distances, non_member_distances],
        }
    )


def distance_membership_auc(
    members: pd.DataFrame,
    non_members: pd.DataFrame,
    released: pd.DataFrame,
    *,
    categorical_columns: list[str] | tuple[str, ...],
) -> float:
    """Return AUC of the negative closest-record-distance membership attack."""

    details = distance_membership_details(
        members,
        non_members,
        released,
        categorical_columns=categorical_columns,
    )
    return float(roc_auc_score(details["label"], details["attack_score"]))


def _row_hashes(frame: pd.DataFrame) -> pd.Series:
    return pd.util.hash_pandas_object(frame.round(EXACT_MATCH_DECIMALS), index=False)
