"""Behaviour of the Integrated Gradients explanation path for NN units."""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ageas.nn import Mixer_Classifier
from ageas.tool import Basic_Clf_Explainer
from tests._fixtures import marker_corpus

CONFIGS = Path(__file__).resolve().parents[1] / 'data' / 'configs'


class MarkerArgmaxModel(nn.Module):
    """Predicts the class whose marker gene (genes ``0..K-1``) is largest."""

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.anchor = nn.Parameter(torch.zeros(1))
        self.criterion = nn.CrossEntropyLoss()

    @property
    def device(self):
        return self.anchor.device

    def forward(self, x):
        return 10 * x.reshape(len(x), -1)[:, :self.n_classes] + self.anchor


class StratifyDatasetTest(unittest.TestCase):

    def test_explained_cells_belong_to_the_requested_class(self):
        corpus = marker_corpus(n_classes=3, n_per_class=5)
        explainer = Basic_Clf_Explainer(
            MarkerArgmaxModel(3),
            dataset=corpus,
            device='cpu',
            n_dataloader_workers=0,
        )

        for limit, expected_count in ((None, 5), (3, 3)):
            with self.subTest(sample_limit=limit):
                loader = explainer.stratify_dataset(
                    class_index=2, sample_limit=limit
                )
                labels = torch.cat([y for _, y in loader])
                self.assertEqual(set(labels.tolist()), {2})
                self.assertEqual(len(labels), expected_count)


class NNExplainTest(unittest.TestCase):
    """An untrained Mixer (BatchNorm + dropout) explained on CPU."""

    def setUp(self):
        torch.manual_seed(0)
        self.corpus = marker_corpus(n_classes=3, n_per_class=6)
        with open(CONFIGS / 'sample_panel' / 'resnet' / 'basic_0.json') as f:
            model_params = json.load(f)['model_params']
        self.model = Mixer_Classifier(
            model_params={
                **model_params,
                'num_classes': 3,
                'len_in': len(self.corpus.features),
                'inplanes': 1,
            }
        )

    def explain(self):
        return self.model.explain(
            dataset=self.corpus,
            device='cpu',
            n_dataloader_workers=0,
            exp_step=4,
            exp_batch_size=6,
        )

    def test_explaining_leaves_the_model_unchanged(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.explain()
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)

    def test_explanations_are_deterministic(self):
        pd.testing.assert_frame_equal(self.explain(), self.explain())

    def test_predict_returns_probabilities(self):
        x = torch.as_tensor(
            np.stack([self.corpus[i][0] for i in range(len(self.corpus))])
        )
        proba = self.model.predict(x)
        self.assertEqual(proba.shape, (len(self.corpus), 3))
        self.assertTrue(abs(proba.sum(axis=1) - 1).max() < 1e-5)

    def test_global_matmul_precision_is_restored(self):
        torch.set_float32_matmul_precision('highest')
        self.explain()
        self.assertEqual(torch.get_float32_matmul_precision(), 'highest')


if __name__ == '__main__':
    unittest.main()
