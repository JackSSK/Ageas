#!/usr/bin/env python3
"""Multinomial Naive Bayes classifier.

Wraps :class:`sklearn.naive_bayes.MultinomialNB` in the Lightning-style
:class:`~ageas.classical.sk_template.Classifier_Template` and uses
``feature_log_prob_`` to produce a class-contrastive feature importance table.
"""
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

from ageas.tool import l1_normalize
from ageas.tool.scores import contrast_classes, score_table

from .sk_template import Classifier_Template, to_2d

# Values used for any key missing from ``model_params``.
DEFAULT_PARAMS = {
    'alpha': 1.0,
    'force_alpha': True,
    'fit_prior': True,
    'class_prior': None,
    'num_class': 2,
}


class MNB_Classifier(Classifier_Template):
    """Lightning wrapper around :class:`sklearn.naive_bayes.MultinomialNB`.

    Uses :meth:`~sklearn.naive_bayes.MultinomialNB.partial_fit` so that the
    model can be trained incrementally over multiple training batches with a
    fixed list of classes provided through ``model_params['num_class']``.
    Prediction, saving and loading come from
    :class:`~ageas.classical.sk_template.Classifier_Template`.

    Attributes:
        model: Underlying :class:`~sklearn.naive_bayes.MultinomialNB`
            instance.
    """

    def __init__(
        self,
        fea_names=None,
        model_params: dict = None,
        **kwargs,
    ) -> None:
        """Initialize an MNB_Classifier.

        Args:
            fea_names: Feature names (gene symbols or Ensembl IDs).
            model_params: Dict of hyper-parameters. Recognized keys:
                ``alpha``, ``force_alpha``, ``fit_prior``, ``class_prior``,
                ``num_class``. Missing keys take their value from
                ``DEFAULT_PARAMS``.
            **kwargs: Forwarded to the parent
                :class:`~ageas.classical.sk_template.Classifier_Template`.
        """
        model_params = {**DEFAULT_PARAMS, **(model_params or {})}

        super().__init__(fea_names=fea_names, model_params=model_params, **kwargs)
        self.model = MultinomialNB(
            alpha=model_params['alpha'],
            force_alpha=model_params['force_alpha'],
            fit_prior=model_params['fit_prior'],
            class_prior=model_params['class_prior'],
        )

    def forward(self, x, y) -> None:
        """Incrementally fit the Naive Bayes model with the current batch.

        Args:
            x: Input feature tensor.
            y: Label tensor.
        """
        self.model.partial_fit(
            to_2d(x),
            np.array(y),
            classes=list(range(self.hparams.model_params['num_class'])),
        )

    def explain(self, score_name: str = 'Scores', **kwargs) -> pd.DataFrame:
        """Class-contrastive feature importance from ``feature_log_prob_``.

        Computes a per-feature contribution magnitude, subtracts the
        cross-class background (max over the other classes), and
        L1-normalises the result per class.

        Args:
            score_name: Unused; kept for API symmetry.
            **kwargs: Ignored extra arguments.

        Returns:
            Score table indexed by feature with columns
            ``Class_{i}_Scores``.
        """
        log_prob = self.model.feature_log_prob_  # (n_classes, n_features)

        contribution = np.mean(np.abs(log_prob), axis=0)
        contribution = contribution / np.sum(contribution)

        contrasted = contrast_classes(log_prob) * contribution
        scores = np.stack(
            [l1_normalize(row, mode='numpy') for row in contrasted]
        )
        return score_table(self.hparams.fea_names, scores)
