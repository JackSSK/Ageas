"""Behaviour of the classical (non-NN) classifier wrappers."""
import unittest

import numpy as np
from torch.utils.data import TensorDataset

from ageas.classical import (
    LogReg_Classifier,
    MNB_Classifier,
    SVM_Classifier,
    XGB_Classifier,
)
from tests._fixtures import (
    LOGREG_PARAMS,
    MNB_PARAMS,
    SVC_PARAMS,
    XGB_PARAMS,
    XGB_TRAIN_CONFIG,
    marker_data,
)


def fit_linear(clf_class, params, n_classes):
    x, y, genes = marker_data(n_classes=n_classes)
    clf = clf_class(
        fea_names=genes,
        model_params={**params, 'num_class': n_classes},
    )
    clf.forward(x, y)
    return clf, x, y


def fit_xgb(n_classes, train_config=XGB_TRAIN_CONFIG):
    x, y, genes = marker_data(n_classes=n_classes)
    clf = XGB_Classifier(
        fea_names=genes,
        model_params={**XGB_PARAMS, 'num_class': n_classes},
        train_config=dict(train_config),
    )
    clf.forward(x, y)
    return clf, x, y


class LinearImportanceTest(unittest.TestCase):

    def test_binary_scores_point_to_their_own_class(self):
        for clf_class, params in (
            (LogReg_Classifier, LOGREG_PARAMS),
            (SVM_Classifier, SVC_PARAMS),
        ):
            with self.subTest(model=clf_class.__name__):
                clf, _, _ = fit_linear(clf_class, params, n_classes=2)
                scores = clf.explain()
                self.assertEqual(scores['Class_0_Scores'].idxmax(), 'g0')
                self.assertEqual(scores['Class_1_Scores'].idxmax(), 'g1')

    def test_multiclass_logreg_scores_are_per_class(self):
        clf, x, _ = fit_linear(LogReg_Classifier, LOGREG_PARAMS, n_classes=3)
        proba_before = clf.predict(x)

        scores = clf.explain()

        for k in range(3):
            column = scores[f'Class_{k}_Scores']
            self.assertEqual(column.idxmax(), f'g{k}')
            self.assertAlmostEqual(column.abs().sum(), 1.0, places=6)
        self.assertTrue((clf.predict(x) == proba_before).all())

    def test_multiclass_svc_explain_is_rejected(self):
        # SVC's multi-class coef_ holds one-vs-one rows, not one row per class.
        clf, _, _ = fit_linear(SVM_Classifier, SVC_PARAMS, n_classes=3)
        with self.assertRaises(NotImplementedError):
            clf.explain()

    def test_mnb_scores_rank_each_class_marker_first(self):
        clf, x, _ = fit_linear(MNB_Classifier, MNB_PARAMS, n_classes=3)
        log_prob_before = clf.model.feature_log_prob_.copy()

        scores = clf.explain()

        for k in range(3):
            column = scores[f'Class_{k}_Scores']
            self.assertEqual(column.idxmax(), f'g{k}')
            self.assertAlmostEqual(column.abs().sum(), 1.0, places=6)
        np.testing.assert_array_equal(
            clf.model.feature_log_prob_, log_prob_before
        )


class XGBTest(unittest.TestCase):

    def test_explain_covers_every_class_when_one_is_absent(self):
        clf, x, y = fit_xgb(n_classes=3)
        keep = y != 0  # explanation data holds classes 1 and 2 only

        scores = clf.explain(dataset=TensorDataset(x[keep], y[keep]))

        self.assertEqual(
            [c for c in scores.columns if c.endswith('_Scores')],
            ['Class_0_Scores', 'Class_1_Scores', 'Class_2_Scores'],
        )
        self.assertEqual(scores['Class_1_Scores'].idxmax(), 'g1')
        self.assertEqual(scores['Class_2_Scores'].idxmax(), 'g2')

    def test_train_config_with_evals_key_trains(self):
        # trainer.xgb_clf's fallback train_config contains 'evals': None.
        clf, x, y = fit_xgb(
            n_classes=2,
            train_config={**XGB_TRAIN_CONFIG, 'evals': None},
        )
        self.assertEqual(clf.predict(x, y).shape, (len(x), 2))


class PredictTest(unittest.TestCase):

    def test_svc_prediction_does_not_depend_on_batch_composition(self):
        clf, x, _ = fit_linear(SVM_Classifier, SVC_PARAMS, n_classes=2)
        whole = clf.predict(x)
        first_ten = clf.predict(x[:10])
        np.testing.assert_allclose(first_ten, whole[:10], rtol=1e-6)

    def test_every_classical_model_predicts_a_single_cell(self):
        fitted = {
            'logreg': fit_linear(LogReg_Classifier, LOGREG_PARAMS, 2),
            'svc': fit_linear(SVM_Classifier, SVC_PARAMS, 2),
            'mnb': fit_linear(MNB_Classifier, MNB_PARAMS, 2),
            'xgb': fit_xgb(2),
        }
        for name, (clf, x, y) in fitted.items():
            with self.subTest(model=name):
                self.assertEqual(clf.predict(x[:1], y[:1]).shape, (1, 2))


class PartialConfigTest(unittest.TestCase):

    def test_missing_keys_fall_back_to_defaults(self):
        # Tutorial 03 writes a logreg config with only these keys.
        partial = {
            LogReg_Classifier: {
                'penalty': 'l2', 'C': 1.0, 'solver': 'lbfgs', 'tol': 1e-4,
            },
            SVM_Classifier: {'C': 1.0, 'kernel': 'linear'},
            MNB_Classifier: {'alpha': 0.5},
        }
        x, y, genes = marker_data(n_classes=2)
        for clf_class, params in partial.items():
            with self.subTest(model=clf_class.__name__):
                clf = clf_class(
                    fea_names=genes,
                    model_params={**params, 'num_class': 2},
                )
                clf.forward(x, y)
                self.assertEqual(clf.predict(x).shape, (len(x), 2))


if __name__ == '__main__':
    unittest.main()
