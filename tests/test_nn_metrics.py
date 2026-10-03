"""NN validation/test metrics are computed over the whole split."""
import tempfile
import unittest

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from ageas.nn import NN_Classifier
from tests._fixtures import marker_corpus


def perfect_model(n_classes, n_genes):
    """A linear NN whose logit k is 10 x gene k: perfect on marker data."""
    model = NN_Classifier(model_params={
        'len_in': n_genes,
        'inplanes': 1,
        'block_nums': [],
        'block_dims': [],
        'latent_fea_dim': None,
        'num_classes': n_classes,
        'dropout': 0.0,
        'norm_layer': 'LayerNorm',
    })
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.zero_()
        for k in range(n_classes):
            model.fc.weight[k, k] = 10.0
    return model


class WholeSplitMetricTest(unittest.TestCase):

    def test_small_batches_do_not_distort_f1_or_auroc(self):
        # Classes are grouped, so batches of 2 mostly hold a single class.
        corpus = marker_corpus(n_classes=3, n_per_class=5)
        model = perfect_model(3, len(corpus.features))
        loader = DataLoader(corpus, batch_size=2, shuffle=False)

        with tempfile.TemporaryDirectory() as folder:
            trainer = pl.Trainer(
                accelerator='cpu', devices=1, logger=False,
                enable_checkpointing=False, enable_progress_bar=False,
                enable_model_summary=False, default_root_dir=folder,
            )
            vali = trainer.validate(model, dataloaders=loader, verbose=False)[0]
            test = trainer.test(model, dataloaders=loader, verbose=False)[0]

        for split, metrics in (('vali', vali), ('test', test)):
            for name in ('accuracy', 'f1', 'auroc'):
                with self.subTest(metric=f'{split}.{name}'):
                    self.assertAlmostEqual(metrics[f'{split}.{name}'], 1.0, places=5)


if __name__ == '__main__':
    unittest.main()
