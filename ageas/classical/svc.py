#!/usr/bin/env python3
"""Support vector machine classifier.

Wraps :class:`sklearn.svm.SVC` in a Lightning-style template, with a
:class:`~sklearn.preprocessing.StandardScaler` applied to the features
before fitting and predicting.

Note:
    Only the ``'linear'`` kernel exposes a usable ``coef_`` attribute, so
    explanations are only meaningful for linear SVMs. Non-linear kernels
    emit a warning at construction time.
"""
import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .sk_template import Classifier_Template, to_2d

# Values used for any key missing from ``model_params``.
DEFAULT_PARAMS = {
    'scaler_copy': True,
    'scaler_with_mean': True,
    'scaler_with_std': True,
    'C': 1.0,
    'kernel': 'linear',
    'degree': 3,
    'gamma': 'scale',
    'coef0': 0.0,
    'shrinking': True,
    'tol': 0.001,
    'cache_size': 200,
    'class_weight': None,
    'verbose': False,
    'max_iter': -1,
    'decision_func_shape': 'ovr',
    'break_ties': False,
    'random_state': None,
    'num_class': 2,
}


class SVM_Classifier(Classifier_Template):
    """Lightning wrapper around :class:`sklearn.svm.SVC` with input scaling.

    Attributes:
        model: Underlying :class:`~sklearn.svm.SVC` instance.
        scaler: :class:`~sklearn.preprocessing.StandardScaler` applied
            before fitting and predicting.
    """

    def __init__(
        self,
        fea_names=None,
        model_params: dict = None,
        **kwargs,
    ) -> None:
        """Initialize an SVM_Classifier.

        Args:
            fea_names: Feature names (gene symbols or Ensembl IDs).
            model_params: Dict of hyper-parameters. Scaler keys:
                ``scaler_copy``, ``scaler_with_mean``, ``scaler_with_std``.
                SVC keys: ``C``, ``kernel``, ``degree``, ``gamma``,
                ``coef0``, ``shrinking``, ``tol``, ``cache_size``,
                ``class_weight``, ``verbose``, ``max_iter``,
                ``decision_func_shape``, ``break_ties``, ``random_state``,
                ``num_class``. Missing keys take their value from
                ``DEFAULT_PARAMS``.
            **kwargs: Forwarded to the parent
                :class:`~ageas.classical.sk_template.Classifier_Template`.
        """
        model_params = {**DEFAULT_PARAMS, **(model_params or {})}

        super().__init__(fea_names=fea_names, model_params=model_params, **kwargs)
        self.model = SVC(
            C=model_params['C'],
            kernel=model_params['kernel'],
            degree=model_params['degree'],
            gamma=model_params['gamma'],
            coef0=model_params['coef0'],
            shrinking=model_params['shrinking'],
            probability=True,
            tol=model_params['tol'],
            cache_size=model_params['cache_size'],
            class_weight=model_params['class_weight'],
            verbose=model_params['verbose'],
            max_iter=model_params['max_iter'],
            decision_function_shape=model_params['decision_func_shape'],
            break_ties=model_params['break_ties'],
            random_state=model_params['random_state'],
        )
        self.scaler = StandardScaler(
            copy=model_params['scaler_copy'],
            with_mean=model_params['scaler_with_mean'],
            with_std=model_params['scaler_with_std'],
        )

        if model_params['kernel'] != 'linear':
            warnings.warn(
                'SVM with non-linear kernel does not support coefficient-based '
                'explanation; explanations will be meaningless.'
            )

    def apply_scaler(self, x, fit: bool = False) -> np.ndarray:
        """Return the standard-scaled features of ``x``.

        Args:
            x: Input feature tensor.
            fit: If ``True``, fit the scaler on ``x`` first. Only training
                data should fit the scaler; prediction reuses its statistics.

        Returns:
            Standard-scaled feature matrix as :class:`numpy.ndarray`.
        """
        if fit:
            self.scaler.fit(to_2d(x))
        return self.scaler.transform(to_2d(x))

    def forward(self, x, y) -> None:
        """Fit the scaler and the underlying SVC on ``(x, y)``.

        Args:
            x: Input feature tensor.
            y: Label tensor.
        """
        self.model.fit(self.apply_scaler(x, fit=True), np.array(y))

    def predict(self, x, y=None) -> np.ndarray:
        """Scale ``x`` with the training statistics and return probabilities.

        Args:
            x: Input feature tensor.
            y: Unused; kept for API symmetry.

        Returns:
            Output of ``self.model.predict_proba`` on the scaled inputs.
        """
        return self.model.predict_proba(self.apply_scaler(x))

    def explain(self, score_name: str = 'Scores', **kwargs):
        """Coefficient-based feature importance (binary problems only).

        Raises:
            NotImplementedError: For more than two classes, where SVC's
                ``coef_`` holds one row per one-vs-one class pair rather than
                one row per class.
        """
        if self.hparams.model_params['num_class'] > 2:
            raise NotImplementedError(
                'SVM_Classifier.explain supports binary problems only: '
                'multi-class SVC coefficients are one-vs-one and cannot be '
                'mapped to per-class scores.'
            )
        return super().explain(score_name=score_name, **kwargs)
