"""Reproducibility and per-round bookkeeping of n_kfold_selection."""
import inspect
import tempfile
import unittest

from ageas import n_kfold_selection
from tests._fixtures import LOGREG_PARAMS, SVC_PARAMS
from tests.test_ops import noisy_marker_corpus, write_hangar

# random_state null: the op's seed has to make these reproducible.
SAGA = {**LOGREG_PARAMS, 'solver': 'saga', 'penalty': 'l1', 'random_state': None}
SVC = {**SVC_PARAMS, 'random_state': None}


def run_selection(units, kfold_selection_list, seed=7):
    with tempfile.TemporaryDirectory() as folder:
        hangar = write_hangar(folder, units)
        deck = n_kfold_selection(
            hangar=hangar,
            query_dataset=noisy_marker_corpus(n_per_class=12),
            kfold_selection_list=kfold_selection_list,
            retention_point=0.0,
            cutoff_point=0.0,
            skip_final=True,
            seed=seed,
            verbose=False,
        )
    return deck


def fold_values(deck, operation='trail'):
    """{(unit, round, split, fold, metric): value} for every recorded fold."""
    values = {}
    for unit_id, unit in deck.squad.items():
        for round_key, record in unit.report[operation].items():
            for split, folds in record.items():
                for i, metrics in enumerate(folds):
                    for name, value in metrics.items():
                        values[(unit_id, round_key, split, i, name)] = float(value)
    return values


class SelectionReproducibilityTest(unittest.TestCase):

    def test_same_seed_gives_identical_results(self):
        units = {'logreg': {'saga': SAGA}, 'svc': {'a': SVC}}
        first = run_selection(units, [2])
        second = run_selection(units, [2])
        self.assertEqual(list(first.squad), list(second.squad))
        self.assertEqual(fold_values(first), fold_values(second))

    def test_each_round_draws_new_folds(self):
        deck = run_selection({'logreg': {'a': LOGREG_PARAMS}}, [2, 2])
        record = deck.squad['logreg_a'].report['trail']
        cel = {
            key: [float(m['test.CEL']) for m in record[key]['test']]
            for key in ('round_1', 'round_2')
        }
        self.assertNotEqual(cel['round_1'], cel['round_2'])


class RoundRecordTest(unittest.TestCase):

    def test_one_record_per_round_with_one_entry_per_fold(self):
        deck = run_selection({'logreg': {'a': LOGREG_PARAMS}}, [2, 3])
        report = deck.squad['logreg_a'].report['trail']
        self.assertEqual(sorted(report), ['round_1', 'round_2'])
        self.assertEqual(len(report['round_1']['test']), 2)
        self.assertEqual(len(report['round_2']['test']), 3)
        self.assertIsNot(report['round_1']['test'], report['round_2']['test'])

    def test_folds_are_stratified_by_default(self):
        params = inspect.signature(n_kfold_selection).parameters
        self.assertIs(params['stratified_kfold_test'].default, True)
        self.assertIs(params['stratified_kfold_valid'].default, True)


if __name__ == '__main__':
    unittest.main()
