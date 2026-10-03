"""Shipped configs load, build, and say what they train with."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from ageas import Hangar
from ageas.config import validate_unit_config
from ageas.tool.trainer import Trainer_Maker
from ageas.unit import SUPPORTED_TYPES
from tests._fixtures import marker_corpus, write_configs

CONFIGS = Path(__file__).resolve().parents[1] / 'data' / 'configs'
PANELS = [p for p in sorted(CONFIGS.iterdir()) if any(c.is_dir() for c in p.iterdir())]
NN_TYPES = ('mlp', 'rnn', 'resnet')


def build_nn(unit_type, config, n_genes=24, n_classes=3):
    model_params = {
        **config['model_params'],
        'num_classes': n_classes, 'len_in': n_genes, 'inplanes': 1,
    }
    train_config = {**config.get('train_config', {}), 'total_steps': 10}
    return SUPPORTED_TYPES[unit_type](
        model_params=model_params, train_config=train_config
    )


class ShippedConfigTest(unittest.TestCase):

    def test_every_panel_loads(self):
        self.assertTrue(PANELS)
        for panel in PANELS:
            with self.subTest(panel=panel.name):
                self.assertTrue(Hangar(config_folder=str(panel)).units)

    def test_loose_config_file_is_valid(self):
        path = CONFIGS / 'mnb' / 'sample_mnb.json'
        validate_unit_config('mnb', json.loads(path.read_text()), source=str(path))

    def test_every_nn_config_builds_and_runs(self):
        x = torch.zeros((4, 1, 24))
        for path in sorted(CONFIGS.glob('*/*/*.json')):
            unit_type = path.parent.name
            if unit_type not in NN_TYPES:
                continue
            with self.subTest(config=f'{path.parts[-3]}/{unit_type}/{path.name}'):
                model = build_nn(unit_type, json.loads(path.read_text()))
                self.assertEqual(tuple(model(x).shape), (4, 3))

    def test_file_name_hyperparameters_are_the_ones_used(self):
        path = next((CONFIGS / 'default_config_v1' / 'mlp').glob('*_lr00001_droupout_02.json'))
        model = build_nn('mlp', json.loads(path.read_text()))
        optimizers = model.configure_optimizers()
        optimizer = optimizers[0][0] if isinstance(optimizers, tuple) else optimizers[0]
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 1e-4)
        self.assertAlmostEqual(model.dropout.p, 0.2)


class UnknownKeyTest(unittest.TestCase):

    def test_typo_is_rejected_with_its_location(self):
        with tempfile.TemporaryDirectory() as folder:
            write_configs(folder, {'logreg': {'typo': {'C': 1.0, 'pnealty': 'l2'}}})
            with self.assertRaisesRegex(ValueError, r"pnealty.*typo\.json"):
                Hangar(config_folder=folder)

    def test_unknown_train_config_key_is_rejected(self):
        config = {'model_params': {'block': 'Residual_Encoder'},
                  'train_config': {'batch_size': 4, 'lr': 1e-4}}
        with self.assertRaisesRegex(ValueError, r"'train_config\.lr'"):
            validate_unit_config('mlp', config, source='inline')


class TrainerSettingsTest(unittest.TestCase):

    def test_lightning_settings_inside_train_config_reach_the_trainer(self):
        corpus = marker_corpus(n_classes=2, n_per_class=4)
        with tempfile.TemporaryDirectory() as folder:
            _, trainer = Trainer_Maker().make(
                model_type='mlp',
                clf_object=SUPPORTED_TYPES['mlp'],
                train_data=corpus, vali_data=corpus, n_classes=2,
                max_epochs=1, dataloader_workers=0, default_root_dir=folder,
                model_params={'block': 'Residual_Encoder', 'block_nums': [1],
                              'block_dims': [4], 'latent_fea_dim': 4,
                              'dropout': 0.0, 'norm_layer': 'LayerNorm'},
                train_config={'batch_size': 4, 'gradient_clip_val': 0.123,
                              'learning_rate': 1e-3},
            )
        self.assertAlmostEqual(trainer.gradient_clip_val, 0.123)


if __name__ == '__main__':
    unittest.main()
