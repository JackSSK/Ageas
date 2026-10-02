"""Behaviour of config loading (Hangar) and device setup (Deck)."""
import tempfile
import unittest
from pathlib import Path

from ageas import Deck, Hangar
from tests._fixtures import LOGREG_PARAMS, write_configs


class DeckTest(unittest.TestCase):

    def test_explicit_cuda_devices_are_kept(self):
        deck = Deck(squad={}, accelerator='cuda', cuda_devices=[0])
        self.assertEqual(list(deck.cuda_devices), [0])


class HangarTest(unittest.TestCase):

    def test_loads_dotted_names_and_skips_hidden_files(self):
        with tempfile.TemporaryDirectory() as folder:
            write_configs(folder, {'logreg': {
                'logreg_C_0.1': LOGREG_PARAMS,
                'logreg_C_0.01': LOGREG_PARAMS,
            }})
            (Path(folder) / '.DS_Store').write_text('')
            (Path(folder) / 'logreg' / '.DS_Store').write_text('')

            hangar = Hangar(config_folder=folder)

        self.assertEqual(
            list(hangar.units),
            ['logreg_logreg_C_0.01', 'logreg_logreg_C_0.1'],
        )


if __name__ == '__main__':
    unittest.main()
