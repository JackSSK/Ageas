#!/usr/bin/env python3
"""N-iteration boost selection.

Iteratively narrows the feature set by repeatedly running
:func:`~ageas.n_kfold_selection` followed by :meth:`~ageas.Deck.debrief`,
keeping only the top features each round. The shrinking schedule is driven
by ``extract_ratio`` and converges toward ``extract_top_n`` features in the
final round.
"""
import logging
from warnings import warn

import pandas as pd

from ageas.hangar import Hangar
from .n_kfold_selection import main as n_kfold_selection
from .n_kfold_selection import monitor_args

_logger = logging.getLogger(__name__)


def main(
    hangar: Hangar,
    operation_name: str = 'boost_selection',
    accelerator: str = 'cpu',
    cuda_devices: list = None,
    query_dataset=None,
    test_dataset=None,
    max_boost_iter: int = 1,
    extract_ratio: float = 0.5,
    extract_top_n: int = 1000,
    seed: int = 42,
    verbose: bool = None,
    selection_args: dict = None,
    explain_args: dict = None,
) -> tuple:
    """N-iteration boost selection driven by per-class explanation scores.

    Each iteration runs :func:`~ageas.n_kfold_selection`, debriefs the
    surviving deck, and trims ``query_dataset`` (and ``test_dataset`` if
    provided) to the top features per class. The number of features kept
    in iteration ``i`` is roughly
    ``extract_top_n / extract_ratio**(max_boost_iter - 1 - i)``,
    converging to ``extract_top_n`` in the final round. After the
    iterative pruning a final selection is run on the trimmed feature set.

    :param hangar: Source hangar from which the operating squad is generated.
    :param operation_name: Operation name under which per-iteration reports
        are stored.
    :param accelerator: Accelerator hint forwarded to selection (``'cpu'``
        or ``'cuda'``).
    :param cuda_devices: Optional GPU device indices.
    :param query_dataset: Required dataset whose feature axis is iteratively
        pruned.
    :param test_dataset: Optional held-out test dataset, pruned in lock-step
        with ``query_dataset``.
    :param max_boost_iter: Number of boost iterations.
    :param extract_ratio: Geometric ratio of feature shrinkage between
        iterations.
    :param extract_top_n: Total number of features kept after the last boost
        iteration.
    :param seed: Random seed forwarded to selection.
    :param verbose: If ``True``, emit per-iteration feature count logs.
    :param selection_args: Extra keyword arguments forwarded to
        :func:`~ageas.n_kfold_selection`.
    :param explain_args: Extra keyword arguments forwarded to
        :meth:`~ageas.Deck.debrief`.
    :returns: ``(deck, fea_keep)`` — the final deck after the post-boost
        selection and the list of feature IDs retained.
    """
    if selection_args is None:
        selection_args = {}
    if explain_args is None:
        explain_args = {}

    assert query_dataset is not None, \
        "Query dataset must be provided for extraction."
    temp_query = query_dataset.copy()
    temp_test = test_dataset.copy() if test_dataset is not None else None

    fea_keep = []
    for i in range(max_boost_iter):
        deck = n_kfold_selection(
            hangar=hangar,
            operation_name=f'{operation_name}_{i+1}',
            accelerator=accelerator,
            cuda_devices=cuda_devices,
            query_dataset=temp_query,
            test_dataset=temp_test,
            seed=seed,
            verbose=verbose,
            **selection_args,
        )
        integrated_exps = deck.debrief(
            operation=f'{operation_name}_{i+1}',
            exp_dataset=temp_query,
            verbose=verbose,
            **{**monitor_args(selection_args), **explain_args},
        )
        if integrated_exps is None:
            warn(f"No valid explanation for iteration {i+1}. Skipping it.")
            continue

        # The feature budget shrinks geometrically to extract_top_n in the
        # last iteration and is split evenly across classes.
        target_nfea = extract_top_n / extract_ratio ** (max_boost_iter - 1 - i)
        n_per_class = int(target_nfea / len(temp_query.label_dict))

        fea_keep = top_features_per_class(integrated_exps, n_per_class)
        _logger.info(
            "Iteration %d — keeping %d features from %d explanations.",
            i + 1,
            len(fea_keep),
            len(integrated_exps.columns),
        )

        temp_query.restrict_features(fea_keep)
        if temp_test is not None:
            temp_test.restrict_features(fea_keep)

    deck = n_kfold_selection(
        hangar=hangar,
        operation_name=f'{operation_name}_final',
        accelerator=accelerator,
        cuda_devices=cuda_devices,
        query_dataset=temp_query,
        test_dataset=temp_test,
        seed=seed,
        verbose=verbose,
        **selection_args,
    )
    return deck, fea_keep


def top_features_per_class(scores: pd.DataFrame, n_per_class: int) -> list:
    """Union of each column's ``n_per_class`` highest-scoring features.

    :param scores: Per-class score table from :meth:`~ageas.Deck.debrief`.
    :param n_per_class: Number of features taken from each column.
    :returns: Feature IDs without duplicates, in order of first appearance
        (column by column, best first).
    """
    keep = {}
    for column in scores.columns:
        best = scores[column].sort_values(ascending=False).head(n_per_class)
        for feature in best.index:
            keep[feature] = None
    return list(keep)
