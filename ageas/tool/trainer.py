#!/usr/bin/env python3
"""Trainer factory for Ageas.

Provides :class:`Trainer_Maker`, a dispatch object that returns a
fully-configured trainer-and-model pair for each supported model type
(scikit-learn, XGBoost, or PyTorch Lightning), and :class:`Fake_Trainer`,
a Lightning-Trainer look-alike that lets non-NN models flow through the
same deck/sortie pipeline.
"""
import logging

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    GradientAccumulationScheduler,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader, Dataset
from torchmetrics import AUROC, Accuracy, F1Score

_logger = logging.getLogger(__name__)


def probability_cross_entropy(
    probs: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Mean cross-entropy of predicted class probabilities.

    Classical models return probabilities, not logits;
    :class:`torch.nn.CrossEntropyLoss` would apply softmax to them again.
    """
    return nn.functional.nll_loss(torch.log(probs.clamp_min(1e-12)), labels)


class Fake_Trainer:
    """Lightning-trainer look-alike for non-NN models.

    Mimics the subset of the :class:`pytorch_lightning.Trainer` API consumed
    by :class:`~ageas.Deck` (``fit``, ``predict``, ``test``,
    ``callback_metrics``) so that scikit-learn and XGBoost classifiers can
    flow through the same sortie pipeline as the Lightning-based models.

    Attributes:
        criterion: Cross-entropy of predicted probabilities
            (:func:`probability_cross_entropy`).
        result_metrics: Per-batch metric dictionaries collected during
            validation and test.
        accuracy: Multiclass accuracy metric.
        f1: Macro-averaged F1 metric.
        auroc: Macro-averaged AUROC metric.
    """

    def __init__(self, n_classes: int = 2, *args, **kwargs) -> None:
        """Initialize a Fake_Trainer.

        Args:
            n_classes: Number of output classes.
            *args: Ignored; accepted for API symmetry with Lightning Trainer.
            **kwargs: Ignored; accepted for API symmetry with Lightning
                Trainer.
        """
        self.criterion = probability_cross_entropy
        self.result_metrics: list = []
        task = 'multiclass'
        self.accuracy = Accuracy(num_classes=n_classes, task=task)
        self.f1 = F1Score(num_classes=n_classes, task=task, average='macro')
        self.auroc = AUROC(num_classes=n_classes, task=task, average='macro')

    def fit(
        self,
        model=None,
        train_dataloaders=None,
        val_dataloaders=None,
        ckpt_path: str = None,
        verbose: bool = False,
    ) -> tuple:
        """Train ``model`` on ``train_dataloaders`` and validate.

        Mimics :meth:`pytorch_lightning.Trainer.fit`.

        Args:
            model: Classifier to train.
            train_dataloaders: Dataloader for the training set.
            val_dataloaders: Dataloader for the validation set.
            ckpt_path: Optional checkpoint path to restore from before
                training.
            verbose: If ``True``, print per-batch progress.

        Returns:
            Tuple ``(train_result, vali_result)``.
        """
        train_result: list = []
        vali_result = None

        if ckpt_path is not None:
            model.load_model(ckpt_path)

        for i, (data, label) in enumerate(train_dataloaders):
            model(data, label)
            if verbose:
                _logger.debug('\tDone model training on batch %d', i)

        if val_dataloaders is not None:
            vali_result = self.predict(model, val_dataloaders)
            self.result_metrics.append(vali_result[-1])

        return train_result, vali_result

    def _score(self, model, dataloaders, split: str, ckpt_path: str = None) -> list:
        """Per-batch metric dicts keyed ``'{split}.CEL'``, ``'{split}.accuracy'``,
        ``'{split}.f1'`` and ``'{split}.auroc'``."""
        if ckpt_path is not None:
            model.load_model(ckpt_path)

        results: list = []
        for data, label in dataloaders:
            preds = torch.Tensor(model.predict(data, label))
            results.append({
                f'{split}.CEL': self.criterion(preds, label),
                f'{split}.accuracy': self.accuracy(preds, label.cpu()),
                f'{split}.f1': self.f1(preds, label.cpu()),
                f'{split}.auroc': self.auroc(preds, label.cpu()),
            })
        return results

    def predict(
        self,
        model=None,
        dataloaders=None,
        ckpt_path: str = None,
        **kwargs,
    ) -> list:
        """Score the validation dataloaders without updating the model.

        Args:
            model: Classifier to evaluate.
            dataloaders: Validation dataloader.
            ckpt_path: Optional checkpoint path to restore from.
            **kwargs: Ignored extra arguments.

        Returns:
            Per-batch list of dicts with ``vali.CEL``, ``vali.accuracy``,
            ``vali.f1`` and ``vali.auroc``.
        """
        return self._score(model, dataloaders, 'vali', ckpt_path)

    def test(
        self,
        model=None,
        dataloaders=None,
        ckpt_path: str = None,
        **kwargs,
    ) -> list:
        """Test the model on ``dataloaders``.

        Args:
            model: Classifier to evaluate.
            dataloaders: Test dataloader.
            ckpt_path: Optional checkpoint path to restore from.
            **kwargs: Ignored extra arguments.

        Returns:
            Per-batch list of dicts with ``test.CEL``, ``test.accuracy``,
            ``test.f1`` and ``test.auroc``. They also become the latest
            :attr:`callback_metrics`.
        """
        test_result = self._score(model, dataloaders, 'test', ckpt_path)
        self.result_metrics.extend(test_result)
        return test_result

    @property
    def callback_metrics(self) -> dict:
        """Most recent metric dict from validation or test."""
        return self.result_metrics[-1]


class Trainer_Maker:
    """Dispatcher that builds a trainer-and-model pair for any supported type.

    The mapping :attr:`SUPPORTED_TYPES` routes each model type to one of
    three concrete factory methods: :meth:`pl_clf` for Lightning-based NN
    classifiers, :meth:`sk_clf` for scikit-learn estimators, and
    :meth:`xgb_clf` for XGBoost.

    Attributes:
        SUPPORTED_TYPES: Dict mapping type key to factory method.
    """

    def __init__(self) -> None:
        self.SUPPORTED_TYPES = {
            'svc': self.sk_clf,
            'logreg': self.sk_clf,
            'mnb': self.sk_clf,
            'xgb': self.xgb_clf,
            'resnet': self.pl_clf,
            'mlp': self.pl_clf,
            'rnn': self.pl_clf,
        }

    def make(self, model_type: str, clf_object, **kwargs) -> tuple:
        """Build the trainer + model pair for ``model_type``.

        Args:
            model_type: Short type key as used in
                :data:`~ageas.unit.SUPPORTED_TYPES`.
            clf_object: Classifier class to instantiate.
            **kwargs: Additional keyword arguments forwarded to the matching
                factory method.

        Returns:
            Tuple ``(model, trainer)`` ready to be fit by the deck.

        Raises:
            ValueError: If ``model_type`` is not in :attr:`SUPPORTED_TYPES`.
        """
        if model_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Model type '{model_type}' is not supported. "
                f"Choose from {list(self.SUPPORTED_TYPES)}."
            )
        return self.SUPPORTED_TYPES[model_type](clf_object=clf_object, **kwargs)

    def pl_clf(
        self,
        clf_object,
        device: list = None,
        accelerator: str = 'cpu',
        dataloader_workers: int = 10,
        max_epochs: int = 1,
        default_root_dir: str = 'cache',
        pretrained_ckpt: str = None,
        enable_progress_bar: bool = False,
        train_data: Dataset = None,
        vali_data: Dataset = None,
        n_classes: int = 2,
        model_params: dict = None,
        train_config: dict = None,
        precision: int = 32,
        num_nodes: int = 1,
        log_every_n_steps: int = 1,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: float = 0.5,
        gradient_clip_algorithm: str = 'value',
        ckpt_every_n_epochs: int = 1,
        save_last: bool = True,
        monitor: str = 'vali.accuracy',
        save_top_k_ckpt: int = 1,
        gradient_accum_schedule: dict = None,
        enable_checkpointing: bool = False,
        **kwargs,
    ) -> tuple:
        """Build a Lightning ``Trainer`` and fit a Lightning classifier.

        Resolves the model's input shape from the first training batch, sets
        ``model_params`` accordingly, optionally loads a pre-trained
        checkpoint, and runs ``trainer.fit``.

        Args:
            clf_object: Lightning classifier class to instantiate.
            device: GPU device list forwarded to the Trainer.
            accelerator: Accelerator string (``'cpu'`` or ``'cuda'``).
            dataloader_workers: Worker count for training/validation
                dataloaders.
            max_epochs: Number of training epochs.
            default_root_dir: Root directory for Lightning logs/checkpoints.
            pretrained_ckpt: Optional path to a pre-trained checkpoint.
            enable_progress_bar: If ``True``, enable Lightning's progress bar.
            train_data: Training dataset.
            vali_data: Validation dataset.
            n_classes: Number of output classes.
            model_params: Architecture hyper-parameters dict.
            train_config: Training hyper-parameters dict.
            precision: Floating-point precision (16 or 32).
            num_nodes: Number of nodes for distributed training.
            log_every_n_steps: Frequency of metric logging.
            accumulate_grad_batches: Gradient accumulation steps.
            gradient_clip_val: Gradient clipping value.
            gradient_clip_algorithm: Clipping algorithm (``'value'`` or
                ``'norm'``).
            ckpt_every_n_epochs: Checkpoint save frequency in epochs.
            save_last: If ``True``, always save the final checkpoint.
            monitor: Metric to monitor for best-checkpoint selection.
            save_top_k_ckpt: Number of top checkpoints to keep.
            gradient_accum_schedule: Optional dict for
                :class:`~pytorch_lightning.callbacks.GradientAccumulationScheduler`.
            enable_checkpointing: If ``True``, enable checkpoint callbacks.
            **kwargs: Ignored extra arguments.

        Returns:
            Tuple ``(model, trainer)`` after the fit call has returned.
        """
        if model_params is None:
            model_params = {}
        if train_config is None:
            train_config = {}

        callbacks: list = []
        if enable_checkpointing:
            callbacks.append(
                ModelCheckpoint(
                    every_n_epochs=ckpt_every_n_epochs,
                    save_last=save_last,
                    monitor=monitor,
                    save_top_k=save_top_k_ckpt,
                )
            )
        if gradient_accum_schedule is not None:
            callbacks.append(
                GradientAccumulationScheduler(scheduling=gradient_accum_schedule)
            )

        trainer = Trainer(
            max_epochs=max_epochs,
            devices=device,
            accelerator=accelerator,
            precision=precision,
            num_nodes=num_nodes,
            log_every_n_steps=log_every_n_steps,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            default_root_dir=default_root_dir,
            enable_checkpointing=enable_checkpointing,
            enable_progress_bar=enable_progress_bar,
            callbacks=callbacks,
        )

        train_dataloader = DataLoader(
            train_data,
            batch_size=train_config['batch_size'],
            shuffle=True,
            num_workers=dataloader_workers,
        )
        val_dataloader = DataLoader(
            vali_data,
            batch_size=train_config['batch_size'],
            shuffle=False,
            num_workers=dataloader_workers,
        )

        sample_data, _ = next(iter(train_dataloader))
        model_params['num_classes'] = n_classes
        model_params['len_in'] = sample_data.shape[-1]
        model_params['inplanes'] = sample_data.shape[-2]

        total_train_steps = int(np.ceil(len(train_dataloader) * max_epochs))
        train_config['total_steps'] = total_train_steps

        if pretrained_ckpt is not None:
            _logger.info("Loading pretrained model from %s", pretrained_ckpt)
            ckpt = torch.load(
                pretrained_ckpt,
                map_location=lambda storage, loc: storage,
            )
            model = clf_object.load_from_checkpoint(
                pretrained_ckpt,
                **ckpt['hyper_parameters'],
            )
            model.to('cpu')
        else:
            model = clf_object(
                model_params=model_params,
                train_config=train_config,
            )

        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=pretrained_ckpt,
        )
        return model, trainer

    def sk_clf(
        self,
        clf_object,
        device: list = None,
        accelerator: str = 'cpu',
        dataloader_workers: int = 10,
        max_epochs: int = 1,
        default_root_dir: str = 'cache',
        pretrained_ckpt: str = None,
        train_data: Dataset = None,
        vali_data: Dataset = None,
        n_classes: int = 2,
        fea_names: list = None,
        model_params: dict = None,
        **kwargs,
    ) -> tuple:
        """Build a :class:`Fake_Trainer` and fit a scikit-learn classifier.

        Args:
            clf_object: scikit-learn classifier class to instantiate.
            device: Ignored for scikit-learn models.
            accelerator: Must be ``'cpu'`` for scikit-learn classifiers.
            dataloader_workers: Worker count for training/validation
                dataloaders.
            max_epochs: Passed to the Fake_Trainer (unused).
            default_root_dir: Passed to the Fake_Trainer (unused).
            pretrained_ckpt: Optional checkpoint path to restore from.
            train_data: Training dataset.
            vali_data: Validation dataset.
            n_classes: Number of output classes.
            fea_names: Feature names forwarded to the classifier.
            model_params: Hyper-parameters dict for the classifier.
            **kwargs: Ignored extra arguments.

        Returns:
            Tuple ``(model, trainer)`` after the fit call has returned.

        Raises:
            ValueError: If ``accelerator`` is not ``'cpu'``.
        """
        if model_params is None:
            model_params = {}

        if accelerator != 'cpu':
            raise ValueError('Only CPU is supported for scikit-learn models.')

        model_params['num_class'] = n_classes
        clf_model = clf_object(
            fea_names=fea_names,
            model_params=model_params,
            **kwargs,
        )

        trainer = Fake_Trainer(
            n_classes=n_classes,
            devices=device,
            accelerator=accelerator,
            max_epochs=max_epochs,
            default_root_dir=default_root_dir,
        )

        trainer.fit(
            model=clf_model,
            train_dataloaders=DataLoader(
                train_data,
                batch_size=len(train_data),
                shuffle=True,
                num_workers=dataloader_workers,
            ),
            val_dataloaders=DataLoader(
                vali_data,
                batch_size=len(vali_data),
                shuffle=False,
                num_workers=dataloader_workers,
            ),
            ckpt_path=pretrained_ckpt,
        )
        return clf_model, trainer

    def xgb_clf(
        self,
        clf_object,
        device: list = None,
        accelerator: str = 'cpu',
        dataloader_workers: int = 10,
        max_epochs: int = 1,
        default_root_dir: str = 'cache',
        pretrained_ckpt: str = None,
        train_data: Dataset = None,
        vali_data: Dataset = None,
        n_classes: int = 2,
        fea_names: list = None,
        model_params: dict = None,
        train_config: dict = None,
        **kwargs,
    ) -> tuple:
        """Build a :class:`Fake_Trainer` and fit an XGBoost classifier.

        Resolves the GPU/CPU device from ``accelerator``, ensures
        ``model_params`` carries ``num_class`` and ``objective``, and runs
        the fit.

        Args:
            clf_object: XGBoost classifier class to instantiate.
            device: GPU device index or list. Used when
                ``accelerator='cuda'``.
            accelerator: ``'cpu'`` or ``'cuda'``.
            dataloader_workers: Worker count for training/validation
                dataloaders.
            max_epochs: Passed to the Fake_Trainer (unused).
            default_root_dir: Passed to the Fake_Trainer (unused).
            pretrained_ckpt: Optional checkpoint path.
            train_data: Training dataset.
            vali_data: Validation dataset.
            n_classes: Number of output classes.
            fea_names: Feature names forwarded to the classifier.
            model_params: XGBoost model parameters dict.
            train_config: XGBoost training parameters dict.
            **kwargs: Ignored extra arguments.

        Returns:
            Tuple ``(model, trainer)`` after the fit call has returned.

        Raises:
            ValueError: If ``accelerator`` is not ``'cpu'`` or ``'cuda'``.
        """
        if model_params is None:
            model_params = {
                'tree_method': 'auto',
                'max_depth': 6,
            }
        if train_config is None:
            train_config = {
                'num_boost_round': 100,
                'evals': None,
                'early_stopping_rounds': None,
                'evals_result': None,
                'verbose_eval': True,
            }

        model_params['num_class'] = n_classes
        if 'objective' not in model_params:
            model_params['objective'] = 'multi:softprob'

        if accelerator == 'cpu':
            model_params['device'] = 'cpu'
        elif accelerator == 'cuda':
            if device is not None:
                if isinstance(device, int):
                    model_params['device'] = f'cuda:{device}'
                elif isinstance(device, list):
                    assert len(device) == 1, 'Only single GPU is supported for XGBoost'
                    model_params['device'] = f'cuda:{device[0]}'
            else:
                model_params['device'] = 'cuda'
        else:
            raise ValueError(
                f"Accelerator '{accelerator}' is not supported for XGBoost. "
                "Choose 'cpu' or 'cuda'."
            )

        clf_model = clf_object(
            fea_names=fea_names,
            model_params=model_params,
            train_config=train_config,
        )

        trainer = Fake_Trainer(
            n_classes=n_classes,
            devices=device,
            accelerator=accelerator,
            max_epochs=max_epochs,
            default_root_dir=default_root_dir,
        )

        trainer.fit(
            model=clf_model,
            train_dataloaders=DataLoader(
                train_data,
                batch_size=len(train_data),
                shuffle=True,
                num_workers=dataloader_workers,
            ),
            val_dataloaders=DataLoader(
                vali_data,
                batch_size=len(vali_data),
                shuffle=False,
                num_workers=dataloader_workers,
            ),
            ckpt_path=pretrained_ckpt,
        )
        return clf_model, trainer
