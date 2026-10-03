#!/usr/bin/env python3
"""N-iteration k-fold model selection.

Runs successive rounds of k-fold cross-validation over a hangar's units,
keeping only the units that survive the configured retention/cutoff thresholds
between rounds. After all rounds the surviving models are retrained on the
full dataset (the "last mission") to produce the final deck used for
downstream prediction or factor extraction.
"""
import copy
import logging
import time

import numpy as np
from pytorch_lightning import seed_everything

from ageas.deck import Deck, last_round, mean_metric
from ageas.hangar import Hangar
from ageas.tool import kfold_random_split
from ageas.tool.corpus_loader import oversample_corpus

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
    means = {
        unit_id: mean_metric(records, monitor_metric)
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


def passes_final_filter(value, retention_point: float, monitor_type: str) -> bool:
    """Whether a unit's out-of-fold metric reaches ``retention_point`` (inclusive)."""
    if monitor_type == 'max':
        return value >= retention_point
    if monitor_type == 'min':
        return value <= retention_point
    return False


def monitor_args(selection_args: dict) -> dict:
    """The selection's monitor settings from a ``selection_args`` dict.

    Passed to :meth:`~ageas.Deck.debrief` so that units are weighted by the
    same metric they were selected on.
    """
    return {
        key: selection_args[key]
        for key in ('monitor_type', 'monitor_metric')
        if key in selection_args
    }


def _require_survivors(deck: Deck, stage: str, **thresholds) -> None:
    """Raise a clear error when a selection stage removed every unit."""
    if not deck.squad:
        settings = ', '.join(f'{name}={value!r}' for name, value in thresholds.items())
        raise ValueError(
            f'No unit survived {stage} ({settings}). Loosen these thresholds, '
            'or check that the units can learn the labels.'
        )


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
    stratified_kfold_test: bool = True,
    stratified_kfold_valid: bool = True,
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

    Each round performs a fresh k-fold cross-validation over the squad,
    ranks the units by ``monitor_metric`` and keeps the top
    ``selection_ratio`` plus any unit above ``retention_point``, while
    discarding everything below ``cutoff_point``. After the rounds, a final
    filter keeps the units whose out-of-fold metric in the last round
    reaches ``retention_point``. The "last mission" then refits them on the
    whole query dataset. The test set is only scored, for reporting: it
    never takes part in selection.

    :param hangar: Source hangar from which the operating squad is generated.
    :param operation_name: Name under which the per-unit reports are stored.
    :param accelerator: Accelerator hint, ``'cpu'`` or ``'cuda'``.
    :param cuda_devices: Optional GPU device indices. ``None`` uses all
        visible devices.
    :param query_dataset: Dataset that is split by k-fold for training and
        validation.
    :param test_dataset: Optional held-out test set, scored in the last
        mission and stored in each unit's ``'final'`` record for reporting.
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
    :param monitor_metric: Dotted metric used to rank units between rounds
        and for the final filter. ``'test.*'`` metrics are the CV test folds,
        i.e. out-of-fold.
    :param retention_point: Units beating this value survive a round
        regardless of rank; in the final filter it is the bar every unit must
        reach (inclusive).
    :param cutoff_point: Hard minimum: units below this value are dropped.
    :param selection_ratio: Fraction of the squad to keep when ranked by the
        metric.
    :param skip_final: If ``True``, skip the final filter and the last
        mission.
    :param seed: Makes the whole run reproducible: seeds Python, NumPy and
        torch at the start, splits round ``r`` (0-based) with ``seed + r``,
        and fills any ``random_state``/XGBoost ``seed`` a unit's config
        leaves unset. ``None`` leaves everything unseeded.
    :param verbose: If ``True``, emit per-fold and per-round progress logs.
    :returns: The deck after selection with only the surviving units. Each
        surviving unit's ``report[operation_name]`` holds ``'round_<r>'``
        entries (per-fold ``'vali'``/``'test'`` metric lists, 1-based ``r``)
        and, unless ``skip_final``, a ``'final'`` entry.
    :raises ValueError: If a round or the final filter leaves no unit; the
        message names the thresholds involved.
    """
    if kfold_selection_list is None:
        kfold_selection_list = [5]
    if seed is not None:
        seed_everything(seed, verbose=False)

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
        seed=seed,
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
            # A different split every round.
            random_seed=None if seed is None else seed + deploy_i,
        )
        assert len(train_list) == len(test_list) == k_fold, \
            "Train and test data lists must have the same length."

        round_report = deck.make_report()
        start_time = time.time()
        for i in range(k_fold):
            _logger.info("Fold %d / %d", i + 1, k_fold)
            deck.sortie(
                report=round_report,
                train_data=train_list[i],
                vali_data=valid_list[i],
                n_classes=n_classes,
                fea_names=fea_names,
                test_dataset=test_list[i],
                save_model=False,
                save_trainer=False,
                verbose=verbose,
            )

        # Units are ranked on every fold recorded so far, across all rounds.
        for unit_id, record in round_report.items():
            for split, folds in record.items():
                deck.report[unit_id][split].extend(folds)

        unit_rank = rank_units(deck.report, monitor_metric, monitor_type)
        deck.squad_update(round_survivors(
            unit_rank,
            expect_survival=expect_survival,
            retention_point=retention_point,
            cutoff_point=cutoff_point,
            monitor_type=monitor_type,
        ))
        _require_survivors(
            deck, f'selection round {deploy_i + 1}',
            monitor_metric=monitor_metric, retention_point=retention_point,
            cutoff_point=cutoff_point,
        )

        for unit_id, unit in deck.squad.items():
            if deploy_i == 0:
                unit.report[operation_name] = dict()
            unit.report[operation_name][f'round_{deploy_i + 1}'] = copy.deepcopy(
                round_report[unit_id]
            )

        _logger.info(
            "Survived: %d  |  elapsed: %.1fs",
            len(deck.squad),
            time.time() - start_time,
        )

    if not skip_final:
        # Final filter: each unit's out-of-fold metric from the last round.
        # The test set plays no part in selection.
        if kfold_selection_list:
            deck.squad_update([
                unit_id
                for unit_id, unit in deck.squad.items()
                if passes_final_filter(
                    mean_metric(
                        unit.report[operation_name][
                            last_round(unit.report[operation_name])
                        ],
                        monitor_metric,
                    ),
                    retention_point,
                    monitor_type,
                )
            ])
            _require_survivors(
                deck, 'the final filter',
                monitor_metric=monitor_metric, retention_point=retention_point,
            )
        else:
            _logger.warning(
                "No selection rounds ran, so the final filter is skipped."
            )

        # Last mission: refit the survivors on all query data, oversampled
        # like the CV training folds. The test set is scored for reporting.
        _logger.info("Last mission start.")
        train_data = query_dataset
        if oversample_method is not None:
            train_data = oversample_corpus(
                query_dataset,
                oversample_method=oversample_method,
                oversample_by=oversample_by,
                random_seed=seed,
            )
        report = deck.sortie(
            report=deck.make_report(),
            train_data=train_data,
            vali_data=query_dataset,
            n_classes=n_classes,
            fea_names=fea_names,
            test_dataset=test_dataset,
            save_model=True,
            save_trainer=True,
            verbose=verbose,
        )

        for unit_id, unit in deck.squad.items():
            unit.report.setdefault(operation_name, {})['final'] = {
                'vali': report[unit_id]['vali'][0],
                'test': report[unit_id]['test'][0],
            }
        _logger.info("Last mission end. Units: %d", len(deck.squad))

    return deck
