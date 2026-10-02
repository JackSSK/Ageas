"""Metrics reported by the trainer used for classical models."""
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ageas.tool.trainer import Fake_Trainer


class FixedProbabilityModel:
    """Predicts probability 0.9 for the true class of every sample."""

    def predict(self, x, y):
        proba = np.full((len(y), 2), 0.1)
        proba[np.arange(len(y)), np.asarray(y)] = 0.9
        return proba


class FakeTrainerTest(unittest.TestCase):

    def test_cross_entropy_is_computed_on_probabilities(self):
        y = torch.tensor([0, 1, 1, 0, 1], dtype=torch.long)
        x = torch.zeros((len(y), 1, 3))
        loader = DataLoader(TensorDataset(x, y), batch_size=len(y))

        result = Fake_Trainer(n_classes=2).test(
            model=FixedProbabilityModel(), dataloaders=loader
        )

        # -log(0.9) for every sample.
        self.assertAlmostEqual(float(result[0]['test.CEL']), 0.1053605, places=5)


if __name__ == '__main__':
    unittest.main()
