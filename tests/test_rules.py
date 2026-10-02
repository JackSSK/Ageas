"""Worked examples for the score-table format and the selection rules."""
import unittest

import numpy as np
import pandas as pd

from ageas.deck import last_round, mean_metric, metric_weight, split_metric_key
from ageas.ops.n_iter_boost_selection import top_features_per_class
from ageas.ops.n_iter_extraction import find_outliers, rank_top_factors
from ageas.ops.n_kfold_selection import passes_final_filter, round_survivors
from ageas.tool.scores import contrast_classes, label_score_columns, score_table


class ScoreTableTest(unittest.TestCase):

    def test_contrast_subtracts_the_best_other_class(self):
        per_class = np.array([[3.0, 0.0], [1.0, 2.0], [0.0, 1.0]])
        np.testing.assert_array_equal(
            contrast_classes(per_class),
            [[2.0, -2.0], [-2.0, 1.0], [-3.0, -1.0]],
        )

    def test_table_layout_and_labels(self):
        table = score_table(['g0', 'g1'], np.ones((2, 2)), stds=np.zeros((2, 2)))
        self.assertEqual(
            list(table.columns),
            ['Class_0_Scores', 'Class_0_Std', 'Class_1_Scores', 'Class_1_Std'],
        )
        scores_only = score_table(['g0', 'g1'], np.ones((2, 2)))
        labelled = label_score_columns(scores_only, {0: 'Ery', 1: 'Neu'})
        self.assertEqual(list(labelled.columns), ['Pro_Ery_Scores', 'Pro_Neu_Scores'])


class SelectionRuleTest(unittest.TestCase):

    RANK = [('a', 0.95), ('b', 0.80), ('c', 0.75), ('d', 0.60)]

    def test_round_keeps_top_ranks_and_retained_units_above_cutoff(self):
        keep = round_survivors(self.RANK, 1, retention_point=0.78,
                               cutoff_point=0.7, monitor_type='max')
        self.assertEqual(keep, ['a', 'b'])

    def test_round_with_lower_is_better_metric(self):
        rank = [('a', 0.1), ('b', 0.3), ('c', 0.6)]
        keep = round_survivors(rank, 1, retention_point=0.35,
                               cutoff_point=0.5, monitor_type='min')
        self.assertEqual(keep, ['a', 'b'])

    def test_fold_mean_of_a_metric(self):
        round_record = {
            'vali': [{'vali.accuracy': 0.2}, {'vali.accuracy': 0.4}],
            'test': [{'test.accuracy': 0.6}, {'test.accuracy': 1.0}],
        }
        self.assertAlmostEqual(mean_metric(round_record, 'test.accuracy'), 0.8)
        final_record = {'vali': {'vali.accuracy': 1.0}, 'test': None}
        self.assertEqual(mean_metric(final_record, 'vali.accuracy'), 1.0)
        self.assertEqual(last_round({'round_2': {}, 'round_10': {}, 'final': {}}),
                         'round_10')
        self.assertTrue(passes_final_filter(0.9, 0.9, 'max'))
        self.assertFalse(passes_final_filter(0.2, 0.1, 'min'))

    def test_metric_weighting(self):
        self.assertEqual(split_metric_key('vali.CEL'), ('vali', 'CEL'))
        self.assertEqual(metric_weight(0.75, 'max'), 0.75)
        # exp(-CEL): the geometric-mean probability given to the true class.
        self.assertAlmostEqual(metric_weight(np.log(2), 'min'), 0.5)
        with self.assertRaises(ValueError):
            metric_weight(0.5, 'MAX')


class FeatureRuleTest(unittest.TestCase):

    def test_outlier_beyond_ten_iqr(self):
        scores = pd.DataFrame(
            {'Class_0_Scores': [0.0, 1.0, 2.0, 3.0, 100.0]},
            index=['a', 'b', 'c', 'd', 'e'],
        )
        # q25 = 1, q75 = 3, so the threshold is 3 + 10 * 2 = 23.
        self.assertEqual(find_outliers(scores), ['e'])

    def test_rank_points_and_per_class_top_features(self):
        scores = pd.DataFrame(
            {'Class_0_Scores': [3.0, 2.0, 1.0], 'Class_1_Scores': [4.0, 0.0, 5.0]},
            index=['x', 'y', 'z'],
        )
        self.assertEqual(top_features_per_class(scores, 1), ['x', 'z'])

        points = rank_top_factors(scores, extract_top_n=2)
        self.assertEqual(points.loc['x', 'Class_0_Scores'], 2)
        self.assertEqual(points.loc['y', 'Class_0_Scores'], 1)
        self.assertEqual(points.loc['z', 'Class_1_Scores'], 2)
        self.assertTrue(pd.isna(points.loc['z', 'Class_0_Scores']))


if __name__ == '__main__':
    unittest.main()
