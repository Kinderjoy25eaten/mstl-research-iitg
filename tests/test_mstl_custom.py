"""
Unit tests for Custom MSTL Algorithm 1 Implementation.

Authors: Rup Narayan Jha, Jaypal Malviya, Anuj Kumar, Ravi Kiran
Mehta Family School of Data Science and AI, IIT Guwahati
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from mstl_custom import MSTLCustom, mstl_decompose, _filter_seasonal_periods, _boxcox, _inv_boxcox


class TestMSTLCustom(unittest.TestCase):

    def setUp(self):
        # Reproducible synthetic signal with 2 known periodicities
        np.random.seed(42)
        self.n = 500
        t = np.arange(self.n)
        self.trend = 100 + 0.05 * t
        self.s24 = 20 * np.sin(2 * np.pi * t / 24)
        self.s168 = 10 * np.cos(2 * np.pi * t / 168)
        self.noise = np.random.normal(0, 2, self.n)
        self.x = self.trend + self.s24 + self.s168 + self.noise

    def test_exact_reconstruction(self):
        """Reconstruction error must be strictly near zero."""
        res = mstl_decompose(self.x, periods=[24, 168], iterations=2)
        reconstructed = res['trend'] + res['seasonal'][24] + res['seasonal'][168] + res['remainder']
        max_err = np.max(np.abs(self.x - reconstructed))
        self.assertLess(max_err, 1e-10)
        self.assertAlmostEqual(res['reconstruction_max_error'], max_err, places=10)

    def test_ascending_period_sorting(self):
        """MSTL must sort periods in ascending order even if passed out of order."""
        model = MSTLCustom(periods=[168, 24])
        self.assertEqual(model.periods, [24, 168])

    def test_period_filtering(self):
        """Periods where 2m >= n must be filtered out."""
        # For n = 100: period 24 has 2*24 = 48 < 100 (kept)
        # period 60 has 2*60 = 120 >= 100 (filtered out)
        valid = _filter_seasonal_periods([24, 60], n=100)
        self.assertEqual(valid, [24])

    def test_missing_value_interpolation(self):
        """NaN values should be linearly interpolated without crashing."""
        x_nan = self.x.copy()
        x_nan[10:15] = np.nan
        x_nan[100] = np.nan
        res = mstl_decompose(x_nan, periods=[24, 168], iterations=2)
        self.assertEqual(res['nan_mask'].sum(), 6)
        self.assertFalse(np.isnan(res['trend']).any())
        self.assertFalse(np.isnan(res['remainder']).any())

    def test_boxcox_inversion(self):
        """Box-Cox forward and inverse transformation must be exact inverses."""
        vals = np.array([10.0, 25.0, 100.0, 500.0])
        for lam in [0.0, 0.5, 1.0, 2.0]:
            trans = _boxcox(vals, lam)
            inv = _inv_boxcox(trans, lam)
            np.testing.assert_allclose(vals, inv, rtol=1e-10)

    def test_single_period_reduction(self):
        """MSTL should work gracefully with a single period."""
        res = mstl_decompose(self.x, periods=[24], iterations=1)
        self.assertIn(24, res['seasonal'])
        self.assertEqual(len(res['seasonal']), 1)
        self.assertLess(res['reconstruction_max_error'], 1e-10)


if __name__ == '__main__':
    unittest.main()
