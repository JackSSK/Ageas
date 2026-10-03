#!/usr/bin/env python3
"""Deck object for Ageas.

The deck operates a squad of units drawn from a :class:`~ageas.Hangar`:
it dispatches sorties, keeps per-unit operation reports, drives prediction,
and aggregates per-class explanation factors via :meth:`Deck.debrief`.
"""
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from warnings import warn

from ageas.tool import Trainer_Maker
from ageas.tool.scores import drop_std_columns, l1_normalize_columns

_logger = logging.getLogger(__name__)


def split_metric_key(metric_key: str) -> tuple:
    """Split a dotted metric key such as ``'vali.CEL'`` into its parts.

    Returns:
        ``(split, name)``, e.g. ``('vali', 'CEL')``. The split is the key
        under which the metric is filed in a report (``'vali'``/``'test'``).
    """
    split, name = metric_key.split('.', 1)
    return split, name


def mean_metric(record: dict, metric_key: str) -> float:
    """Mean of ``metric_key`` over the folds of a report record.

    Args:
        record: ``{'vali': ..., 'test': ...}`` where each split holds a list
            of per-fold metric dicts (a ``'round_<r>'`` record or a deck
            report) or a single metric dict (a ``'final'`` record).
        metric_key: Dotted metric key, e.g. ``'test.accuracy'``.
    """
    split, _ = split_metric_key(metric_key)
    folds = record[split]
    if isinstance(folds, dict):
        folds = [folds]
    return float(np.mean([float(fold[metric_key]) for fold in folds]))


def metric_weight(value: float, monitor_type: str) -> float:
    """A unit's unnormalised ``debrief`` weight from its metric value.

    Higher-is-better metrics (``'max'``, e.g. accuracy) are used as they
    are. Lower-is-better metrics (``'min'``, e.g. CEL) become ``exp(-x)``,
    which for CEL is the geometric-mean probability given to the true class.

    Raises:
        ValueError: If ``monitor_type`` is not ``'min'`` or ``'max'``.
    """
    if monitor_type == 'max':
        return float(value)
    if monitor_type == 'min':
        return float(np.exp(-value))
    raise ValueError(f"monitor_type must be 'min' or 'max', got {monitor_type!r}")


def last_round(operation_report: dict) -> str:
    """Key of the last selection round (``'round_<r>'``) in a unit's report."""
    rounds = [key for key in operation_report if key.startswith('round_')]
    if not rounds:
        raise ValueError(
            "The unit has no 'round_<r>' records; run n_kfold_selection first."
        )
    return max(rounds, key=lambda key: int(key.split('_')[1]))


class Deck:
    """Sortie deck that trains, evaluates, and explains a squad of units.

    Operates on a squad produced by :meth:`~ageas.Hangar.sortie_generate`:
    a :class:`~ageas.tool.Trainer_Maker` launches the correct trainer per
    model type, a per-unit report ledger collects metrics, and
    :meth:`debrief` combines model-wise explanations weighted by their
    validation/test metrics.

    :ivar squad: Mapping ``unit_id -> Unit`` for the active operating units.
    :ivar trainer_maker: Factory that produces trainer-model pairs.
    :ivar n_dataloader_workers: Worker count for prediction/test dataloaders.
    :ivar report: Per-unit ``{'vali': [], 'test': []}`` metric ledger.
    :ivar accelerator: Default accelerator (``'cpu'`` or ``'cuda'``).
    :ivar cuda_devices: Resolved GPU device indices (``None`` for CPU).
    """

    def __init__(
        self,
        squad: dict,
        n_dataloader_workers: int = 10,
        accelerator: str = 'cpu',
        cuda_devices: list = None,
        seed: int = None,
    ) -> None:
        """Initialize a Deck.

        :param squad: Mapping ``unit_id -> Unit`` to operate on, typically
            produced by :meth:`~ageas.Hangar.sortie_generate`.
        :param n_dataloader_workers: Number of worker processes for the
            prediction/test dataloaders.
        :param accelerator: Default accelerator (``'cpu'`` or ``'cuda'``) the
            deck will request from each unit at sortie time.
        :param cuda_devices: Optional list of GPU device indices to bind. When
            ``accelerator='cuda'`` and this is ``None``, all visible devices
            are used.
        :param seed: Passed to every trainer, which uses it for model
            parameters such as ``random_state`` that the config leaves unset.
        """
        self.seed = seed
        self.squad = squad
        self.trainer_maker = Trainer_Maker()
        self.n_dataloader_workers = n_dataloader_workers
        self.report = self.make_report(squad)

        self.accelerator = accelerator
        if accelerator != 'cuda':
            self.cuda_devices = None
        elif cuda_devices is not None:
            self.cuda_devices = cuda_devices
        else:
            self.cuda_devices = range(torch.cuda.device_count())
            assert len(self.cuda_devices) > 0, "No CUDA devices found"

    def sortie(
        self,
        train_data: Dataset,
        vali_data: Dataset,
        n_classes: int,
        fea_names: pd.Index,
        test_dataset: Dataset = None,
        report: dict = None,
        save_model: bool = False,
        save_trainer: bool = False,
        verbose: bool = True,
    ) -> dict:
        """Train every unit in the squad on a single fold and collect metrics.

        Each unit is "spiked" to verify its accelerator, then trained via
        the appropriate trainer (Lightning, scikit-learn, or XGBoost).
        Validation metrics are read from the trainer's callback metrics and
        an optional test pass is launched on ``test_dataset``.

        :param train_data: Training split for this sortie.
        :param vali_data: Validation split for this sortie.
        :param n_classes: Number of label classes in the task.
        :param fea_names: Feature names (typically gene symbols/Ensembl IDs).
        :param test_dataset: Optional held-out test split. If provided, each
            model is also evaluated on it after training.
        :param report: Existing report dict to append to. ``None`` uses the
            deck's own ``self.report`` ledger.
        :param save_model: If ``True``, retain the trained model on the unit.
        :param save_trainer: If ``True``, retain the trainer object on the unit.
        :param verbose: If ``True``, print per-unit training and evaluation
            traces.
        :returns: The updated report mapping ``unit_id`` to lists of
            validation and test metric snapshots.
        """
        report = self.report if report is None else report

        for unit_id, unit in self.squad.items():
            if verbose:
                _logger.info("\tType:%s, Unit:%s", unit.type, unit.tail)

            clf_object = unit.spike(
                accelerator=self.accelerator,
                available_devices=self.cuda_devices,
            )
            if clf_object is None:
                warn(f"Remove {unit_id} from the sortie squad.")
                continue

            model, trainer = self.trainer_maker.make(
                model_type=unit.type,
                clf_object=clf_object,
                device=unit.device,
                accelerator=unit.accelerator,
                vali_data=vali_data,
                train_data=train_data,
                n_classes=n_classes,
                fea_names=fea_names,
                seed=self.seed,
                **unit.config,
            )

            vali_result = trainer.callback_metrics

            if test_dataset is not None:
                test_batch_size = len(test_dataset)
                if (
                    'train_config' in unit.config
                    and 'batch_size' in unit.config['train_config']
                ):
                    temp = unit.config['train_config']['batch_size']
                    if temp is not None:
                        test_batch_size = temp

                test_result = trainer.test(
                    model=model,
                    dataloaders=DataLoader(
                        test_dataset,
                        num_workers=self.n_dataloader_workers,
                        batch_size=test_batch_size,
                        shuffle=False,
                    ),
                    verbose=verbose,
                )[0]
            else:
                test_result = None

            if verbose:
                _logger.info("\t\tValidation: %s", vali_result)
                _logger.info("\t\tTest: %s", test_result)

            self.squad[unit_id].model = (
                model if save_model else self.squad[unit_id].model
            )
            self.squad[unit_id].trainer = (
                trainer if save_trainer else self.squad[unit_id].trainer
            )
            report[unit_id]['vali'].append(vali_result)
            report[unit_id]['test'].append(test_result)

        return report

    def squad_update(self, unit_ids: list) -> None:
        """Restrict the squad and report ledger to a subset of unit IDs.

        :param unit_ids: Whitelist of unit IDs to keep. All other units are
            dropped from both ``self.squad`` and ``self.report``.
        """
        self.squad = {k: v for k, v in self.squad.items() if k in unit_ids}
        self.report = {k: v for k, v in self.report.items() if k in unit_ids}

    def make_report(self, squad: dict = None) -> dict:
        """Initialize an empty report ledger for a squad of units.

        :param squad: Mapping ``unit_id -> Unit`` for which to allocate report
            slots. ``None`` uses the deck's current squad.
        :returns: A nested dict ``{unit_id: {'vali': [], 'test': []}}``.
        """
        squad = self.squad if squad is None else squad
        return {
            unit_id: {'vali': [], 'test': []}
            for unit_id in squad
        }

    @property
    def ops_reports(self) -> dict:
        """Aggregate per-operation reports across the entire squad.

        Walks the per-unit ``unit.report`` ledger and reshapes it into a
        nested structure::

            {operation: {mission: {metric_type: {metric_name: {unit_id: [values]}}}}}

        that is convenient for downstream summary plots.

        :returns: The merged report tree described above.
        """
        ans: dict = {}

        def _process_list(parent, met_value, unit_id):
            for exp_rec in met_value:
                if exp_rec is None:
                    continue
                for met_name, met_val in exp_rec.items():
                    parent.setdefault(met_name, {})
                    parent[met_name].setdefault(unit_id, [])
                    parent[met_name][unit_id].append(float(met_val.item()))

        def _process_dict(parent, met_value, unit_id):
            for met_name, met_val in met_value.items():
                parent.setdefault(met_name, {})
                parent[met_name][unit_id] = float(met_val)

        for unit_id, unit in self.squad.items():
            for opt_name, report in unit.report.items():
                ans.setdefault(opt_name, {})
                for mission, result in report.items():
                    ans[opt_name].setdefault(mission, {})
                    for met_type, met_value in result.items():
                        if met_value is None:
                            continue
                        ans[opt_name][mission].setdefault(met_type, {})
                        if isinstance(met_value, dict):
                            _process_dict(
                                ans[opt_name][mission][met_type],
                                met_value,
                                unit_id,
                            )
                        elif isinstance(met_value, list):
                            _process_list(
                                ans[opt_name][mission][met_type],
                                met_value,
                                unit_id,
                            )
                        if len(ans[opt_name][mission][met_type]) == 0:
                            del ans[opt_name][mission][met_type]
        return ans

    def predict(
        self,
        query_dataset: Dataset = None,
        batch_size: int = 10,
        num_workers: int = 1,
    ) -> tuple:
        """Ensemble-predict by averaging probabilities across the squad.

        Each unit produces softmax outputs on ``query_dataset`` which are
        averaged uniformly across all units in the squad.

        :param query_dataset: Dataset to score. Each item must yield
            ``(x, y)`` tuples.
        :param batch_size: Batch size for the prediction dataloader.
        :param num_workers: Number of dataloader workers.
        :returns: Tuple ``(all_preds, all_labels)`` where ``all_preds`` is the
            averaged probability matrix and ``all_labels`` is the concatenated
            label tensor in the same order.
        """
        dataloader = DataLoader(
            query_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        all_preds: list = []
        all_labels: list = []
        for unit_id, unit in self.squad.items():
            for i, (x, y) in enumerate(dataloader):
                pred = unit.model.predict(x, y) / len(self.squad)
                if len(all_preds) <= i:
                    all_preds.append(pred)
                    all_labels.append(y)
                else:
                    all_preds[i] += pred

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        return all_preds, all_labels

    def debrief(
        self,
        exp_dataset: Dataset = None,
        operation: str = 'trail',
        mission: str = None,
        monitor_type: str = 'max',
        monitor_metric: str = 'test.accuracy',
        verbose: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Integrate per-class explanation scores across the squad.

        Each unit's score table is L1-normalised per column, so no unit
        counts more because of its scores' scale. The tables are then
        averaged with weights from :func:`metric_weight`, applied to the
        unit's out-of-fold ``monitor_metric`` and normalised to sum to 1.

        :param exp_dataset: Dataset to explain. Forwarded unchanged to every
            unit's ``explain`` method.
        :param operation: Operation name used to look up the metric in
            ``unit.report``.
        :param mission: Record in the operation's report that supplies the
            metric. ``None`` uses the last selection round (``'round_<r>'``),
            whose metrics are out-of-fold.
        :param monitor_type: ``'max'`` for higher-is-better metrics,
            ``'min'`` for lower-is-better ones (see :func:`metric_weight`).
        :param monitor_metric: Dotted metric name (e.g. ``'test.accuracy'``,
            the out-of-fold accuracy). Use the selection's monitor metric.
        :param verbose: If ``True``, print per-unit weights and the assembled
            answer.
        :param kwargs: Additional keyword arguments forwarded to each
            ``unit.model.explain`` call.
        :returns: The integrated per-class score table, without ``_Std``
            columns.
        :raises ValueError: If the squad is empty.
        """
        if not self.squad:
            raise ValueError('debrief needs at least one unit in the squad.')

        weights = {}
        for unit_id, unit in self.squad.items():
            op_report = unit.report[operation]
            record = op_report[mission if mission is not None else last_round(op_report)]
            weights[unit_id] = metric_weight(
                mean_metric(record, monitor_metric), monitor_type
            )
        total = sum(weights.values())

        general_ans = None
        for unit_id, unit in self.squad.items():
            # All-zero metrics (e.g. accuracy 0 everywhere) fall back to equal weights.
            weight = weights[unit_id] / total if total > 0 else 1 / len(weights)
            if verbose:
                _logger.info("Explaining Unit: %s, Weight: %.4f", unit_id, weight)

            report = unit.model.explain(
                dataset=exp_dataset,
                device=unit.accelerator,
                **kwargs,
            )
            ans = l1_normalize_columns(drop_std_columns(report)) * weight
            general_ans = ans if general_ans is None else general_ans + ans

            if verbose:
                _logger.info("Unit: %s\n%s", unit_id, ans)

        if verbose:
            _logger.info("General Answer\n%s", general_ans)

        return general_ans
