#!/usr/bin/env python3
"""Per-class feature score tables.

Every explainer in Ageas returns the same table: one row per feature and,
for each class ``k``, a ``Class_{k}_Scores`` column (followed by a
``Class_{k}_Std`` column when the explainer measures spread). This module
owns that format, and the class-contrastive step the explainers share.
"""
import numpy as np
import pandas as pd


def score_column(class_index: int) -> str:
    """Name of the score column for ``class_index``."""
    return f'Class_{class_index}_Scores'


def std_column(class_index: int) -> str:
    """Name of the standard-deviation column for ``class_index``."""
    return f'Class_{class_index}_Std'


def class_of_column(column: str) -> int:
    """Class index encoded in a ``Class_{k}_...`` column name."""
    return int(column.split('_')[1])


def contrast_classes(per_class: np.ndarray) -> np.ndarray:
    """Subtract from each class row the maximum of the other class rows.

    Uses the max (aggressive); a mean or median background would give
    softer contrasts.

    Args:
        per_class: Array of shape ``(n_classes, n_features)``; needs at
            least two classes.

    Returns:
        New array of the same shape and dtype; the input is not modified.
    """
    per_class = np.asarray(per_class)
    contrasted = np.empty_like(per_class)
    for k in range(len(per_class)):
        background = np.max(np.delete(per_class, k, axis=0), axis=0)
        contrasted[k] = per_class[k] - background
    return contrasted


def score_table(features, scores, stds=None) -> pd.DataFrame:
    """Build the per-class score table.

    Args:
        features: Feature names, used as the index.
        scores: Array of shape ``(n_classes, n_features)``.
        stds: Optional array of the same shape. When given, each
            ``Class_{k}_Scores`` column is followed by ``Class_{k}_Std``.

    Returns:
        :class:`~pandas.DataFrame` indexed by ``features``.
    """
    columns = {}
    for k in range(len(scores)):
        columns[score_column(k)] = scores[k]
        if stds is not None:
            columns[std_column(k)] = stds[k]
    return pd.DataFrame(columns, index=features)


def drop_std_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Return ``table`` without its standard-deviation columns."""
    return table.drop(columns=[c for c in table.columns if 'Std' in str(c)])


def label_score_columns(table: pd.DataFrame, label_dict: dict) -> pd.DataFrame:
    """Rename ``Class_{k}_Scores`` columns to ``Pro_{label}_Scores``.

    Args:
        table: Table whose columns are all ``Class_{k}_...`` columns.
        label_dict: Mapping from class index to label.
    """
    return table.rename(columns={
        column: f'Pro_{label_dict[class_of_column(column)]}_Scores'
        for column in table.columns
    })
