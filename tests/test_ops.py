"""Behaviour of the selection / extraction operations (logreg units only)."""
import tempfile
import unittest

import numpy as np

from ageas import Hangar, n_iter_extraction, n_kfold_selection
from tests._fixtures import LOGREG_PARAMS, corpus_from_matrix, write_configs


def write_hangar(folder, units):
    write_configs(folder, units)
    return Hangar(config_folder=str(folder))


def noisy_marker_corpus(n_classes=3, n_per_class=10, n_noise=12, seed=0):
    """Gene ``k`` marks class ``k``; the remaining genes are Gaussian noise."""
    rng = np.random.RandomState(seed)
    y = np.repeat(np.arange(n_classes), n_per_class)
    x = rng.normal(0.0, 0.3, size=(len(y), n_classes + n_noise))
    x[np.arange(len(y)), y] += 3.0
    return corpus_from_matrix(x, [f'type_{k}' for k in y])


GOOD = {**LOGREG_PARAMS, 'C': 10.0}
BAD = {**LOGREG_PARAMS, 'C': 1e-6}  # near-uniform probabilities


class KfoldSelectionTest(unittest.TestCase):

    def test_min_monitor_keeps_the_lower_loss_unit(self):
        with tempfile.TemporaryDirectory() as folder:
            hangar = write_hangar(folder, {'logreg': {'good': GOOD, 'bad': BAD}})
            deck = n_kfold_selection(
                hangar=hangar,
                query_dataset=noisy_marker_corpus(),
                kfold_selection_list=[2],
                monitor_type='min',
                monitor_metric='test.CEL',
                retention_point=0.01,
                cutoff_point=0.9,
                selection_ratio=0.5,
                skip_final=True,
                verbose=False,
            )
        self.assertEqual(list(deck.squad), ['logreg_good'])


class ExtractionTest(unittest.TestCase):

    def run_extraction(self, hangar, corpus, exp_dataset=None):
        return n_iter_extraction(
            hangar=hangar,
            query_dataset=corpus,
            exp_dataset=exp_dataset,
            max_extraction_iter=1,
            extract_top_n=2,
            use_gene_names=True,
            verbose=False,
            selection_args={'kfold_selection_list': [2]},
        )

    def test_held_out_genes_are_reported_by_name(self):
        corpus = noisy_marker_corpus()
        with tempfile.TemporaryDirectory() as folder:
            hangar = write_hangar(folder, {'logreg': {'good': GOOD}})
            top_factors, _ = self.run_extraction(hangar, corpus)

        self.assertTrue((top_factors['Outlier_Iter'] >= 0).any())
        self.assertTrue(all(g.startswith('name_') for g in top_factors.index))

    def test_rank_points_and_held_out_scores_have_separate_columns(self):
        corpus = noisy_marker_corpus()
        with tempfile.TemporaryDirectory() as folder:
            hangar = write_hangar(folder, {'logreg': {'good': GOOD}})
            top_factors, _ = self.run_extraction(hangar, corpus)

        labels = ['type_0', 'type_1', 'type_2']
        self.assertEqual(
            sorted(top_factors.columns),
            sorted([f'Pro_{x}_Scores' for x in labels]
                   + [f'Pro_{x}_HeldOut' for x in labels] + ['Outlier_Iter']),
        )
        held = top_factors['Outlier_Iter'] >= 0
        rank_cols = [f'Pro_{x}_Scores' for x in labels]
        held_cols = [f'Pro_{x}_HeldOut' for x in labels]
        self.assertTrue(held.any() and (~held).any())
        self.assertTrue(top_factors.loc[held, rank_cols].isna().all().all())
        self.assertTrue(top_factors.loc[~held, held_cols].isna().all().all())
        # Held-out features come first.
        self.assertTrue(held.iloc[:held.sum()].all())

    def test_user_exp_dataset_is_not_modified(self):
        corpus = noisy_marker_corpus()
        exp_dataset = corpus.copy()
        n_genes = exp_dataset.adata.n_vars
        with tempfile.TemporaryDirectory() as folder:
            hangar = write_hangar(folder, {'logreg': {'good': GOOD}})
            self.run_extraction(hangar, corpus, exp_dataset=exp_dataset)

        self.assertEqual(exp_dataset.adata.n_vars, n_genes)


if __name__ == '__main__':
    unittest.main()
