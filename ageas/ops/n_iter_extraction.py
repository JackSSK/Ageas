#!/usr/bin/env python3
"""N-iteration extraction of top regulatory factors.

Wraps :func:`~ageas.n_kfold_selection` and feeds its surviving deck through
:meth:`~ageas.Deck.debrief` to obtain per-class explanation scores.
Outliers are pruned between iterations and the per-class top-N factors are
ranked across iterations to yield the final regulatory factor table.
"""
import logging
from warnings import warn

import pandas as pd

from ageas.hangar import Hangar
from ageas.tool import l1_normalize
from ageas.tool.scores import label_score_columns, score_column
from .n_kfold_selection import main as n_kfold_selection

_logger = logging.getLogger(__name__)


def main(
    hangar: Hangar,
    operation_name: str = 'extraction',
    accelerator: str = 'cpu',
    cuda_devices: list = None,
    query_dataset=None,
    test_dataset=None,
    exp_dataset=None,
    max_extraction_iter: int = 1,
    extract_top_n: int = 10,
    use_gene_names: bool = True,
    seed: int = 42,
    verbose: bool = None,
    selection_args: dict = None,
    explain_args: dict = None,
) -> tuple:
    """N-iteration extraction of top regulatory factors per class.

    Each iteration runs :func:`~ageas.n_kfold_selection`, debriefs the
    surviving deck to obtain per-class explanation scores, prunes
    outlier features, and feeds the trimmed dataset back into the next
    iteration. The per-iteration explanations are L1-aggregated and the
    top-N factors per class are ranked across iterations.

    :param hangar: Source hangar from which the operating squad is generated.
    :param operation_name: Operation name under which per-iteration reports
        are stored.
    :param accelerator: Accelerator hint forwarded to selection (``'cpu'``
        or ``'cuda'``).
    :param cuda_devices: Optional GPU device indices.
    :param query_dataset: Required dataset that drives selection and (by
        default) the explanation pass.
    :param test_dataset: Optional held-out test dataset.
    :param exp_dataset: Optional explanation dataset. Defaults to
        ``test_dataset`` when provided, otherwise ``query_dataset``.
    :param max_extraction_iter: Number of extraction iterations.
    :param extract_top_n: Number of top factors to retain per class in the
        final ranking.
    :param use_gene_names: If ``True``, the returned table uses
        ``adata.var['name']`` as the index instead of feature IDs.
    :param seed: Random seed forwarded to selection.
    :param verbose: If ``True``, emit per-iteration progress logs.
    :param selection_args: Extra keyword arguments forwarded to
        :func:`~ageas.n_kfold_selection`.
    :param explain_args: Extra keyword arguments forwarded to
        :meth:`~ageas.Deck.debrief`.
    :returns: ``(top_factors, final_exp)``. ``top_factors`` is the per-class
        ranked factor table (with an ``Outlier_Iter`` column flagging
        held-out features), and ``final_exp`` is the L1-normalised
        integrated explanation table.
    """
    if selection_args is None:
        selection_args = {}
    if explain_args is None:
        explain_args = {}

    assert query_dataset is not None, \
        "Query dataset must be provided for extraction."

    temp_query = query_dataset.copy()
    temp_test = test_dataset.copy() if test_dataset is not None else None
    if exp_dataset is None:
        exp_dataset = temp_test if temp_test is not None else temp_query
    else:
        # Features are pruned from it in place below; leave the caller's intact.
        exp_dataset = exp_dataset.copy()

    answers = []
    hold_out = {}
    integrated_exps = None

    for i in range(max_extraction_iter):
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
        _logger.info("Iteration %d — selected %d units.", i + 1, len(deck.squad))

        if len(deck.squad) == 0:
            _logger.warning("No units survived. Stopping extraction.")
            break

        integrated_exps = deck.debrief(
            operation=f'{operation_name}_{i+1}',
            exp_dataset=exp_dataset,
            verbose=verbose,
            **explain_args,
        )
        if integrated_exps is None:
            warn(f"No valid explanation for iteration {i+1}. Skipping it.")
            continue

        assert exp_dataset is not None, "Explain dataset becomes None"

        # Hold out this iteration's outliers: record their scores and the
        # iteration, then drop them from every dataset.
        outliers = find_outliers(integrated_exps)
        for feature in outliers:
            hold_out[feature] = (
                integrated_exps.loc[feature, :].values.tolist() + [i]
            )
        if outliers:
            temp_query.restrict_features(
                ~temp_query.adata.var.index.isin(outliers)
            )
        remaining = temp_query.adata.var.index
        if temp_test is not None:
            temp_test.restrict_features(remaining)
        exp_dataset.restrict_features(remaining)

        if len(remaining) == 0:
            warn("No features left in the query dataset. Stopping extraction.")
            break

        answers.append(integrated_exps)

    final_exp = aggregate_iterations(answers, integrated_exps, temp_query.label_dict)
    top_factors = rank_top_factors(final_exp, extract_top_n)

    top_factors = label_score_columns(top_factors, temp_query.label_dict)
    final_exp = label_score_columns(final_exp, temp_query.label_dict)

    top_factors['Outlier_Iter'] = -1
    for feature, row in hold_out.items():
        top_factors.loc[feature, :] = row

    if use_gene_names:
        # Look names up in the caller's dataset: held-out outliers are no
        # longer in ``temp_query``, which was pruned between iterations.
        assert 'name' in query_dataset.adata.var.columns, \
            "Gene names are not available in the query dataset."
        top_factors.index = [
            query_dataset.adata.var.loc[gene, 'name']
            for gene in top_factors.index
        ]
    return top_factors, final_exp


def find_outliers(
    exp_df: pd.DataFrame,
    tail: str = 'upper',
    iqr_factor: float = 10,
) -> list:
    """Features that are IQR outliers in any column of ``exp_df``.

    A feature is an outlier in a column when its score is at or beyond the
    upper (or lower) quartile by ``iqr_factor`` interquartile ranges. The
    comparison is inclusive, so with a zero IQR every score at or beyond the
    quartile counts.

    :param exp_df: Per-class explanation table from
        :meth:`~ageas.Deck.debrief`.
    :param tail: ``'upper'`` or ``'lower'``; any other value finds nothing.
    :param iqr_factor: IQR multiplier defining the outlier threshold.
    :returns: Outlier feature IDs without duplicates, in order of first
        appearance (column by column).
    """
    assert iqr_factor >= 0, "Outlier IQR factor must be non-negative."
    found = {}
    for col in exp_df.columns:
        q25, q75 = exp_df[col].quantile(0.25), exp_df[col].quantile(0.75)
        distance = iqr_factor * (q75 - q25)
        if tail.upper() == 'UPPER':
            is_outlier = exp_df[col] >= q75 + distance
        elif tail.upper() == 'LOWER':
            is_outlier = exp_df[col] <= q25 - distance
        else:
            continue
        for feature in exp_df.index[is_outlier]:
            found[feature] = None
    return list(found)


def aggregate_iterations(
    answers: list, last_table: pd.DataFrame, label_dict: dict
) -> pd.DataFrame:
    """Sum each class's scores over the kept iterations, then L1-normalise.

    The sum covers the features of ``last_table`` (the latest debrief
    table). Iteration tables are added as they are, without normalising
    each one first.

    :param answers: Debrief tables of the iterations that were kept.
    :param last_table: The latest debrief table; supplies index and columns.
    :param label_dict: Class index to label mapping of the query dataset.
    """
    final_exp = last_table.copy() * 0.0
    for c in label_dict:
        column = score_column(c)
        for exp_df in answers:
            final_exp[column] += exp_df.loc[final_exp.index, column]
        final_exp[column] = l1_normalize(final_exp[column], mode='numpy')
    return final_exp


def rank_top_factors(final_exp: pd.DataFrame, extract_top_n: int) -> pd.DataFrame:
    """Rank points for each class's ``extract_top_n`` highest-scoring features.

    In each column the best feature gets ``extract_top_n`` points (or as many
    as there are features), the next one point less, down to 1. Features
    outside a class's top N are NaN in that class's column.
    """
    top_factors = pd.DataFrame(columns=final_exp.columns)
    for col in final_exp.columns:
        head = final_exp[col].sort_values(ascending=False).head(extract_top_n)
        for points, gene in zip(range(len(head), 0, -1), head.index):
            top_factors.loc[gene, col] = points
    return top_factors
