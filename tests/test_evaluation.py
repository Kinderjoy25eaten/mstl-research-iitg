"""
Unit tests for Evaluation Metrics.

Authors: Rup Narayan Jha, Jaypal Malviya, Anuj Kumar, Ravi Kiran
Mehta Family School of Data Science and AI, IIT Guwahati
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from evaluation import rmse, mae, mape, component_wise_rmse, reconstruction_error


class TestEvaluation(unittest.TestCase):

    def test_rmse_known_values(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 20.0, 34.0])
        # diffs: [2, 0, 4], squares: [4, 0, 16] -> mean = 20/3 -> sqrt(20/3) ≈ 2.581988897
        expected = np.sqrt(20.0 / 3.0)
        self.assertAlmostEqual(rmse(y_true, y_pred), expected, places=7)

    def test_mae_known_values(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 20.0, 34.0])
        expected = 6.0 / 3.0
        self.assertAlmostEqual(mae(y_true, y_pred), expected, places=7)

    def test_component_wise_rmse(self):
        true_dict = {'trend': np.array([1.0, 2.0]), 'daily': np.array([3.0, 4.0])}
        est_dict = {'trend': np.array([1.0, 2.0]), 'daily': np.array([3.5, 4.5])}
        res = component_wise_rmse(true_dict, est_dict)
        self.assertEqual(res['trend'], 0.0)
        self.assertAlmostEqual(res['daily'], 0.5)


if __name__ == '__main__':
    unittest.main()
