#!/usr/bin/env python3
"""N-iteration k-fold model selection.

Runs successive rounds of k-fold cross-validation over a hangar's units,
keeping only the units that survive the configured retention/cutoff thresholds
between rounds. After all rounds the surviving models are retrained on the
full dataset (the "last mission") to produce the final deck used for
downstream prediction or factor extraction.
"""
import logging
import time

import numpy as np

from ageas.deck import Deck, split_metric_key
from ageas.hangar import Hangar
from ageas.tool import kfold_random_split

_logger = logging.getLogger(__name__)


def rank_units(report: dict, monitor_metric: str, monitor_type: str) -> list:
    """Rank units, best first, by their mean metric over one round's folds.

    Args:
        report: Deck report ``{unit_id: {'vali': [...], 'test': [...]}}``
            holding one metric dict per fold.
        monitor_metric: Dotted metric key, e.g. ``'test.accuracy'``.
        monitor_type: ``'max'`` if higher is better, ``'min'`` if lower is.

    Returns:
        List of ``(unit_id, mean_metric)`` pairs, best first.
    """
    split, _ = split_metric_key(monitor_metric)
    means = {
        unit_id: np.mean([float(rec[monitor_metric]) for rec in records[split]])
        for unit_id, records in report.items()
    }
    return sorted(means.items(), key=lambda x: x[1], reverse=monitor_type == 'max')


def _beats(value, threshold, monitor_type: str) -> bool:
    """Strictly better than ``threshold``; 'min' metrics are better when lower."""
    if monitor_type == 'min':
        return value < threshold
    return value > threshold


def round_survivors(
    unit_rank: list,
    expect_survival: int,
    retention_point: float,
    cutoff_point: float,
    monitor_type: str,
) -> list:
    """Unit IDs kept after one selection round.

    A unit survives if it ranks within the top ``expect_survival`` or beats
    ``retention_point``, and in either case beats ``cutoff_point``.

    Args:
        unit_rank: Output of :func:`rank_units`, best first.
    """
    return [
        unit_id
        for rank, (unit_id, value) in enumerate(unit_rank)
        if (rank < expect_survival or _beats(value, retention_point, monitor_type))
        and _beats(value, cutoff_point, monitor_type)
    ]


def final_selection_metric(final_record: dict, monitor_metric: str):
    """Value the last-mission filter compares with ``retention_point``.

    Only the metric name is taken from ``monitor_metric``. The split is the
    test split whenever the last mission had a test set, and otherwise the
    last mission's validation split, which is the full query data the unit
    was trained on.

    Args:
        final_record: ``{'vali': metrics, 'test': metrics or None}``.
        monitor_metric: Dotted metric key, e.g. ``'test.accuracy'``.
    """
    _, name = split_metric_key(monitor_metric)
    if final_record['test'] is not None:
        return final_record['test'][f'test.{name}']
    return final_record['vali'][f'vali.{name}']


def passes_final_filter(value, retention_point: float, monitor_type: str) -> bool:
    """Whether a unit reaches ``retention_point`` (inclusive) after the last mission."""
    if monitor_type == 'max':
        return value >= retention_point
    if monitor_type == 'min':
        return value <= retention_point
    return False


def main(
    hangar: Hangar,
    operation_name: str = 'trail',
    accelerator: str = 'cpu',
    cuda_devices: list = None,
    query_dataset=None,
    test_dataset=None,
    n_dataloader_workers: int = 1,
    using_model_types: list = None,
    using_model_list: list = None,
    kfold_selection_list: list = None,
    valid_fraction: float = 0.1,
    stratified_kfold_test: bool = False,
    stratified_kfold_valid: bool = False,
    oversample_method: str = None,
    oversample_by: str = 'median',
    monitor_type: str = 'max',
    monitor_metric: str = 'test.accuracy',
    retention_point: float = 0.9,
    cutoff_point: float = 0.7,
    selection_ratio: float = 0.5,
    skip_final: bool = False,
    seed: int = 42,
    verbose: bool = None,
) -> 'Deck':
    """N-iteration k-fold selection of top-performing models.

    Each iteration performs a fresh k-fold cross-validation over the squad,
    ranks the units by ``monitor_metric`` and keeps the top
    ``selection_ratio`` plus any unit above ``retention_point``, while
    discarding everything below ``cutoff_point``. After all iterations a
    final pass (the "last mission") retrains the survivors on the full
    dataset and applies a final retention filter.

    :param hangar: Source hangar from which the operating squad is generated.
    :param operation_name: Name under which the per-unit reports are stored.
    :param accelerator: Accelerator hint, ``'cpu'`` or ``'cuda'``.
    :param cuda_devices: Optional GPU device indices. ``None`` uses all
        visible devices.
    :param query_dataset: Dataset that is split by k-fold for training and
        validation.
    :param test_dataset: Optional held-out test set used in the last mission.
    :param n_dataloader_workers: Number of dataloader workers for the deck.
    :param using_model_types: Whitelist of model types to include from the
        hangar.
    :param using_model_list: Whitelist of explicit unit IDs to include from
        the hangar.
    :param kfold_selection_list: Fold counts per selection round. ``[5]`` runs
        a single 5-fold round; ``[2, 3, 4]`` would run three rounds. Defaults
        to ``[5]``.
    :param valid_fraction: Fraction of each training fold reserved for
        validation.
    :param stratified_kfold_test: If ``True``, the test fold is
        class-stratified.
    :param stratified_kfold_valid: If ``True``, the validation split is
        class-stratified.
    :param oversample_method: Oversampling method (e.g. ``'repeat'``).
        ``None`` disables oversampling.
    :param oversample_by: Target class size when oversampling: ``'mean'``,
        ``'median'``, or ``'max'``.
    :param monitor_type: ``'max'`` for higher-is-better metrics, ``'min'``
        for lower-is-better.
    :param monitor_metric: Dotted metric used to rank units between rounds.
    :param retention_point: Units at or above this metric value are retained
        unconditionally.
    :param cutoff_point: Hard minimum: units below this value are dropped.
    :param selection_ratio: Fraction of the squad to keep when ranked by the
        metric.
    :param skip_final: If ``True``, skip the last mission retraining pass.
    :param seed: Random seed forwarded to the k-fold splitter.
    :param verbose: If ``True``, emit per-fold and per-round progress logs.
    :returns: The deck after selection with only the surviving units.
    """
    if kfold_selection_list is None:
        kfold_selection_list = [5]

    n_classes = len(query_dataset.label_dict)
    fea_names = query_dataset.adata.var.index.tolist()

    deck = Deck(
        squad=hangar.sortie_generate(
            unit_types=using_model_types,
            unit_list=using_model_list,
        ),
        n_dataloader_workers=n_dataloader_workers,
        accelerator=accelerator,
        cuda_devices=cuda_devices,
    )
    assert len(deck.squad) > 0, "Not enough models in the squad."

    for deploy_i, k_fold in enumerate(kfold_selection_list):
        expect_survival = max(int(len(deck.squad) * selection_ratio), 1)
        _logger.info(
            "Deployment %d / %d  |  k=%d  |  units=%d",
            deploy_i + 1,
            len(kfold_selection_list),
            k_fold,
            len(deck.squad),
        )

        train_list, valid_list, test_list = kfold_random_split(
            query_dataset,
            n_splits=k_fold,
            valid_fraction=valid_fraction,
            stratified_test=stratified_kfold_test,
            stratified_valid=stratified_kfold_valid,
            oversample_method=oversample_method,
            oversample_by=oversample_by,
            random_seed=seed,
        )
        assert len(train_list) == len(test_list) == k_fold, \
            "Train and test data lists must have the same length."

        for i in range(k_fold):
            start_time = time.time()
            _logger.info("Fold %d / %d", i + 1, k_fold)

            report = deck.sortie(
                train_data=train_list[i],
                vali_data=valid_list[i],
                n_classes=n_classes,
                fea_names=fea_names,
                test_dataset=test_list[i],
                save_model=False,
                save_trainer=False,
                verbose=verbose,
            )

        unit_rank = rank_units(report, monitor_metric, monitor_type)
        deck.squad_update(round_survivors(
            unit_rank,
            expect_survival=expect_survival,
            retention_point=retention_point,
            cutoff_point=cutoff_point,
            monitor_type=monitor_type,
        ))

        for unit_id, unit in deck.squad.items():
            if deploy_i == 0:
                unit.report[operation_name] = dict()
            unit.report[operation_name][f'fold_{i+1}'] = report[unit_id]

        _logger.info(
            "Survived: %d  |  elapsed: %.1fs",
            len(deck.squad),
            time.time() - start_time,
        )

    if not skip_final:
        _logger.info("Last mission start.")
        report = deck.sortie(
            report=deck.make_report(),
            train_data=query_dataset,
            vali_data=query_dataset,
            n_classes=n_classes,
            fea_names=fea_names,
            test_dataset=test_dataset,
            save_model=True,
            save_trainer=True,
            verbose=verbose,
        )

        final_list = []
        for unit_id, unit in deck.squad.items():
            if operation_name not in unit.report:
                unit.report[operation_name] = dict()
            final_record = {
                'vali': report[unit_id]['vali'][0],
                'test': report[unit_id]['test'][0],
            }
            unit.report[operation_name]['final'] = final_record

            metric = final_selection_metric(final_record, monitor_metric)
            if passes_final_filter(metric, retention_point, monitor_type):
                final_list.append(unit_id)
        deck.squad_update(final_list)
        _logger.info("Last mission end. Survivors: %d", len(deck.squad))

    return deck
