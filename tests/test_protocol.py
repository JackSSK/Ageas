"""The test set is for reporting only; debrief weights are normalised."""
import tempfile
import unittest

import numpy as np
import pandas as pd

from ageas import n_iter_extraction, n_kfold_selection
from tests._fixtures import LOGREG_PARAMS, XGB_PARAMS
from tests.test_ops import noisy_marker_corpus, write_hangar

GOOD = {**LOGREG_PARAMS, 'C': 10.0}
WEAK = {**LOGREG_PARAMS, 'C': 0.05}


def scrambled(corpus, seed=0):
    """Copy of ``corpus`` with its labels randomly permuted."""
    copy = corpus.copy()
    labels = copy.adata.obs['celltype']
    copy.adata.obs['celltype'] = pd.Categorical(
        np.random.RandomState(seed).permutation(labels.to_numpy()),
        categories=labels.cat.categories,
    )
    return copy


def select(units, test_dataset, **kwargs):
    with tempfile.TemporaryDirectory() as folder:
        hangar = write_hangar(folder, units)
        return n_kfold_selection(
            hangar=hangar,
            query_dataset=noisy_marker_corpus(n_per_class=12),
            test_dataset=test_dataset,
            kfold_selection_list=[3],
            verbose=False,
            **kwargs,
        )


class TestSetRoleTest(unittest.TestCase):

    def test_scrambled_test_labels_do_not_change_survivors(self):
        test_set = noisy_marker_corpus(n_per_class=12, seed=1)
        units = {'logreg': {'good': GOOD, 'weak': WEAK}}
        honest = select(units, test_set, oversample_method='repeat')
        noisy = select(units, scrambled(test_set), oversample_method='repeat')
        self.assertTrue(honest.squad)
        self.assertEqual(list(honest.squad), list(noisy.squad))

    def test_extraction_ignores_the_test_set(self):
        query = noisy_marker_corpus(n_per_class=12)
        args = dict(max_extraction_iter=2, extract_top_n=3, verbose=False,
                    selection_args={'kfold_selection_list': [2],
                                    'retention_point': 0.0, 'cutoff_point': 0.0})
        with tempfile.TemporaryDirectory() as folder:
            # XGB's SHAP explanation depends on the data it explains.
            hangar = write_hangar(folder, {
                'logreg': {'good': GOOD}, 'xgb': {'a': XGB_PARAMS},
            })
            without = n_iter_extraction(hangar=hangar, query_dataset=query, **args)
            with_test = n_iter_extraction(
                hangar=hangar, query_dataset=query,
                test_dataset=scrambled(noisy_marker_corpus(n_per_class=12, seed=1)),
                **args,
            )
        for a, b in zip(without, with_test):
            pd.testing.assert_frame_equal(a, b)


class ScaledExplainer:
    """Wraps a model so that its explanation scores are multiplied."""

    def __init__(self, model, factor):
        self.model, self.factor = model, factor

    def explain(self, **kwargs):
        return self.model.explain(**kwargs) * self.factor


class DebriefWeightTest(unittest.TestCase):

    def test_a_units_score_scale_does_not_change_the_result(self):
        deck = select({'logreg': {'good': GOOD, 'weak': WEAK}}, None,
                      retention_point=0.0, cutoff_point=0.0)
        query = noisy_marker_corpus(n_per_class=12)
        before = deck.debrief(exp_dataset=query, verbose=False)
        unit = deck.squad['logreg_weak']
        unit.model = ScaledExplainer(unit.model, 100.0)
        after = deck.debrief(exp_dataset=query, verbose=False)
        pd.testing.assert_frame_equal(before, after)


class EmptySquadTest(unittest.TestCase):

    def test_impossible_retention_point_raises_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, 'retention_point'):
            select({'logreg': {'good': GOOD}}, None,
                   retention_point=1.5, cutoff_point=0.0)

    def test_impossible_cutoff_raises_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, 'cutoff_point'):
            select({'logreg': {'good': GOOD}}, None,
                   retention_point=0.0, cutoff_point=1.5)


if __name__ == '__main__':
    unittest.main()
