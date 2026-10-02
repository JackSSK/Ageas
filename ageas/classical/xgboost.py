#!/usr/bin/env python3
"""XGBoost gradient-boosted tree classifier.

Lightning-style wrapper around :func:`xgboost.train` whose ``explain``
method delegates to :class:`shap.TreeExplainer` for SHAP-value based
feature attribution.
"""
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import shap
import torch
import xgboost as xgb
from torch.utils.data import DataLoader

from ageas.tool import l1_normalize
from ageas.tool.scores import contrast_classes, score_table

from .sk_template import to_2d


class XGB_Classifier(pl.LightningModule):
    """Lightning wrapper around XGBoost's :func:`xgboost.train`.

    Stores the trained ``Booster`` as ``self.model`` and an optional
    :class:`shap.TreeExplainer` as ``self.explainer`` for downstream
    feature attribution.

    Attributes:
        model: Trained XGBoost :class:`~xgboost.Booster` (``None`` before
            first forward pass).
        explainer: Cached :class:`shap.TreeExplainer` (``None`` until
            :meth:`make_explainer` is called).
    """

    def __init__(
        self,
        fea_names=None,
        model_params: dict = None,
        train_config: dict = None,
    ) -> None:
        """Initialize an XGB_Classifier.

        Args:
            fea_names: Feature names (gene symbols or Ensembl IDs).
            model_params: XGBoost model parameters forwarded to
                :func:`xgboost.train`. Recognized keys include
                ``booster``, ``tree_method``, ``objective``,
                ``num_class``, ``max_depth``, ``device``.
            train_config: Training configuration forwarded to
                :func:`xgboost.train`. Recognized keys include
                ``num_boost_round``, ``early_stopping_rounds``,
                ``verbose_eval``.
        """
        if model_params is None:
            model_params = {
                'booster': 'gbtree',
                'tree_method': 'auto',
                'objective': 'multi:softprob',
                'num_class': 2,
                'max_depth': 6,
                'device': 'cpu',
            }
        if train_config is None:
            train_config = {
                'num_boost_round': 100,
                'early_stopping_rounds': None,
                'verbose_eval': True,
            }

        super().__init__()
        self.model = None
        self.explainer = None
        self.save_hyperparameters()

    def forward(self, x, y) -> None:
        """Build a ``DMatrix`` from ``(x, y)`` and call :func:`xgboost.train`.

        Re-uses ``self.model`` as ``xgb_model`` so that successive forward
        calls perform incremental boosting. Configures early stopping based
        on ``train_config['early_stopping_rounds']`` when present.

        Args:
            x: Input feature tensor.
            y: Label tensor.
        """
        dtrain = xgb.DMatrix(
            to_2d(x),
            label=np.array(y),
            feature_names=self.hparams.fea_names,
        )

        evals = None
        if (
            'early_stopping_rounds' in self.hparams.train_config
            and self.hparams.train_config['early_stopping_rounds'] is not None
        ):
            evals = [(dtrain, 'train')]
            # Auto-set early stopping rounds if a boolean True was given
            if isinstance(
                self.hparams.train_config['early_stopping_rounds'], bool
            ) and self.hparams.train_config['early_stopping_rounds']:
                self.hparams.train_config['early_stopping_rounds'] = (
                    self.hparams.train_config['num_boost_round'] // 10
                )

        # ``evals`` is decided above; a config-level 'evals' entry would be
        # passed twice.
        train_kwargs = {
            k: v for k, v in self.hparams.train_config.items() if k != 'evals'
        }
        self.model = xgb.train(
            params=self.hparams.model_params,
            dtrain=dtrain,
            xgb_model=self.model,
            evals=evals,
            **train_kwargs,
        )

    def predict(self, x, y) -> np.ndarray:
        """Score ``x`` with the trained Booster and return probabilities.

        Args:
            x: Input feature tensor.
            y: Label tensor (used only to build the DMatrix).

        Returns:
            Probability matrix of shape ``(batch, num_class)``.
        """
        self.model.set_param({'device': self.hparams.model_params['device']})
        return self.model.predict(
            xgb.DMatrix(
                to_2d(x),
                label=np.array(y),
                feature_names=self.hparams.fea_names,
            )
        )

    def make_explainer(
        self,
        background_data=None,
        store_explainer: bool = True,
        device=None,
        **kwargs,
    ) -> shap.TreeExplainer:
        """Build a SHAP :class:`~shap.TreeExplainer` for the current Booster.

        Args:
            background_data: Optional background dataset for the tree
                explainer.
            store_explainer: If ``True``, cache the explainer on
                ``self.explainer``.
            device: Optional XGBoost device override (e.g. ``'cpu'``,
                ``'cuda:0'``) applied to the Booster before constructing
                the explainer.
            **kwargs: Additional keyword arguments forwarded to
                :class:`shap.TreeExplainer`.

        Returns:
            The constructed :class:`shap.TreeExplainer`.
        """
        if device is not None:
            self.model.set_param({'device': device})

        explainer = shap.TreeExplainer(
            self.model,
            background_data,
            feature_perturbation='tree_path_dependent',
            **kwargs,
        )

        self.explainer = explainer if store_explainer else None
        return explainer

    def explain(
        self,
        dataset=None,
        background_data=None,
        batch_size: int = None,
        explainer: shap.TreeExplainer = None,
        device: str = 'cpu',
        **kwargs,
    ) -> pd.DataFrame:
        """Compute per-class SHAP-based feature importance over ``dataset``.

        Iterates the dataset in batches, accumulates per-class SHAP means
        and standard deviations, and produces a class-contrastive importance
        table by subtracting the cross-class background.

        Args:
            dataset: Dataset that yields ``(x, y)`` tuples.
            background_data: Optional background dataset used when
                constructing a fresh explainer.
            batch_size: Batch size for the explanation dataloader. ``None``
                uses the entire dataset as a single batch.
            explainer: Pre-built SHAP explainer. ``None`` builds one via
                :meth:`make_explainer`.
            device: Device override forwarded to :meth:`make_explainer`.
            **kwargs: Additional keyword arguments forwarded to
                :meth:`make_explainer`.

        Returns:
            Per-feature, per-class score and standard-deviation table as a
            :class:`~pandas.DataFrame`.
        """
        loader = DataLoader(
            dataset,
            batch_size=len(dataset) if batch_size is None else batch_size,
            shuffle=False,
        )

        if explainer is None:
            explainer = (
                self.explainer
                if self.explainer is not None
                else self.make_explainer(
                    device=(
                        self.hparams.model_params['device']
                        if device is None
                        else device
                    ),
                    background_data=background_data,
                )
            )

        # Rows are indexed by class id, so size them by the model's classes,
        # not by the labels that happen to be present in ``dataset``.
        n_class = self.hparams.model_params['num_class']
        explain_means = None
        explain_stds = None

        for data, label in loader:
            if explain_means is None:
                explain_means = np.zeros((n_class, data.shape[-1]))
                explain_stds = np.zeros((n_class, data.shape[-1]))

            for q_class in label.unique():
                ct_data = data[label == q_class]
                if len(ct_data) == 0:
                    continue
                explain_result = explainer.shap_values(
                    np.array(ct_data.squeeze(-2, -1))
                )
                # TreeExplainer returns shape (n_samples, n_features, n_classes)
                explain_result = explain_result[:, :, q_class]

                explain_means[q_class] += l1_normalize(
                    np.mean(explain_result, axis=0), mode='numpy'
                )
                explain_stds[q_class] += l1_normalize(
                    np.std(explain_result, axis=0), mode='numpy'
                )

        return score_table(
            self.hparams.fea_names,
            contrast_classes(explain_means),
            stds=explain_stds,
        )

    def save_model(self, path: str) -> None:
        """Save the trained Booster to ``path`` using XGBoost's native format.

        Args:
            path: Destination file path.
        """
        self.model.save_model(path)

    def load_model(self, path: str) -> None:
        """Load a Booster from ``path`` (XGBoost native format).

        Args:
            path: Path to the saved Booster file.
        """
        self.model = xgb.Booster(model_file=path)

    def trans_preds(self, preds: np.ndarray, num_class: int) -> np.ndarray:
        """Convert per-class probabilities into discrete class labels via argmax.

        Args:
            preds: Probability matrix from :meth:`predict`.
            num_class: Number of classes (unused; kept for API symmetry).

        Returns:
            ``argmax`` along the class axis.
        """
        return preds.argmax(axis=-1)
