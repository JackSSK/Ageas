"""Behaviour of the AnnData corpus, splitting and the fake-data helper."""
import unittest

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_selection import f_classif

from ageas.tool import Multimodal_Corpus, kfold_random_split, make_fake_adata


def id_corpus(class_sizes, n_genes=4):
    """Corpus whose gene 0 holds each cell's row number, grouped by class."""
    labels = np.repeat(
        [f'type_{k}' for k in range(len(class_sizes))], class_sizes
    )
    n_cells = len(labels)
    x = np.zeros((n_cells, n_genes), dtype=np.float32)
    x[:, 0] = np.arange(n_cells)
    adata = ad.AnnData(
        csr_matrix(x),
        obs=pd.DataFrame(index=[f'cell_{i}' for i in range(n_cells)]),
        var=pd.DataFrame(index=[f'gene_{j}' for j in range(n_genes)]),
    )
    adata.obs['celltype'] = pd.Categorical(labels)
    return Multimodal_Corpus(adata=adata, label_key='celltype', backed=False)


class StratifyTest(unittest.TestCase):

    def test_stratified_cells_keep_their_class_id(self):
        corpus = id_corpus([5, 5, 5])

        strat = corpus.stratify(class_label=2)

        self.assertEqual(len(strat), 5)
        self.assertEqual({int(strat[i][1]) for i in range(len(strat))}, {2})


def cell_ids(tensor_corpus):
    return set(tensor_corpus.data[:, 0, 0].long().tolist())


class OversampleTest(unittest.TestCase):

    def split(self, corpus, seed):
        return kfold_random_split(
            corpus,
            n_splits=2,
            valid_fraction=0.1,
            oversample_method='repeat',
            oversample_by='max',
            random_seed=seed,
        )

    def test_repeat_keeps_every_original_training_cell(self):
        corpus = id_corpus([50, 45])
        train_list, valid_list, test_list = self.split(corpus, seed=0)

        for train, valid, test in zip(train_list, valid_list, test_list):
            originals = (
                set(range(len(corpus))) - cell_ids(valid) - cell_ids(test)
            )
            self.assertEqual(cell_ids(train), originals)

    def test_repeat_is_reproducible_with_a_seed(self):
        corpus = id_corpus([50, 45])
        first = self.split(corpus, seed=0)[0]
        second = self.split(corpus, seed=0)[0]
        for a, b in zip(first, second):
            self.assertTrue((a.data == b.data).all())


class StratifiedSplitTest(unittest.TestCase):

    def test_too_small_validation_split_falls_back_to_unstratified(self):
        # 15 training cells per fold, 10%: 2 validation cells for 3 classes.
        corpus = id_corpus([10, 10, 10])
        with self.assertLogs('ageas.tool.corpus_loader', level='WARNING'):
            train_list, valid_list, _ = kfold_random_split(
                corpus, n_splits=2, valid_fraction=0.1,
                stratified_test=True, stratified_valid=True, random_seed=0,
            )
        self.assertEqual([len(v) for v in valid_list], [2, 2])


class FakeAdataTest(unittest.TestCase):

    def test_informative_genes_are_the_class_separating_ones(self):
        adata = make_fake_adata(n_cells=200, n_class=2, n_clusters_per_class=1)
        f_stat, _ = f_classif(adata.X.toarray(), adata.obs['celltype'])
        f_stat = pd.Series(f_stat, index=adata.var.index)

        # A single informative axis may carry no class difference (cluster
        # centres can differ along one axis only), so compare the best genes.
        is_noise = f_stat.index.str.startswith('fake_gene_')
        signal, noise = f_stat[~is_noise], f_stat[is_noise]
        self.assertEqual(len(noise), 16)
        self.assertGreater(signal.max(), 10 * noise.max())


if __name__ == '__main__':
    unittest.main()
