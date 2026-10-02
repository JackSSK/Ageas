#!/usr/bin/env python3
"""Logistic regression classifier.

Wraps :class:`sklearn.linear_model.LogisticRegression` in the
Lightning-style :class:`~ageas.classical.sk_template.Classifier_Template`.
"""
from sklearn.linear_model import LogisticRegression

from .sk_template import Classifier_Template

# Values used for any key missing from ``model_params``.
DEFAULT_PARAMS = {
    'penalty': 'l1',
    'dual': False,
    'tol': 1e-4,
    'C': 1.0,
    'fit_intercept': True,
    'intercept_scaling': 1.0,
    'class_weight': None,
    'random_state': None,
    'solver': 'lbfgs',
}


class LogReg_Classifier(Classifier_Template):
    """Lightning wrapper around :class:`sklearn.linear_model.LogisticRegression`.

    Attributes:
        model: Underlying :class:`~sklearn.linear_model.LogisticRegression`
            instance.
    """

    def __init__(
        self,
        fea_names=None,
        model_params: dict = None,
        **kwargs,
    ) -> None:
        """Initialize a LogReg_Classifier.

        Args:
            fea_names: Feature names (gene symbols or Ensembl IDs).
            model_params: Dict of hyper-parameters forwarded to
                :class:`~sklearn.linear_model.LogisticRegression`. Recognized
                keys: ``penalty``, ``dual``, ``tol``, ``C``,
                ``fit_intercept``, ``intercept_scaling``, ``class_weight``,
                ``random_state``, ``solver``. Missing keys take their value
                from ``DEFAULT_PARAMS``.
            **kwargs: Forwarded to the parent
                :class:`~ageas.classical.sk_template.Classifier_Template`.
        """
        model_params = {**DEFAULT_PARAMS, **(model_params or {})}

        super().__init__(fea_names=fea_names, model_params=model_params, **kwargs)
        self.model = LogisticRegression(
            penalty=model_params['penalty'],
            dual=model_params['dual'],
            tol=model_params['tol'],
            C=model_params['C'],
            fit_intercept=model_params['fit_intercept'],
            intercept_scaling=model_params['intercept_scaling'],
            class_weight=model_params['class_weight'],
            random_state=model_params['random_state'],
            solver=model_params['solver'],
        )
