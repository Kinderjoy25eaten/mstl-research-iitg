"""
Unit tests for Data Processing and Victoria Electricity Dataset slicing.

Authors: Rup Narayan Jha, Jaypal Malviya, Anuj Kumar, Ravi Kiran
Mehta Family School of Data Science and AI, IIT Guwahati
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data_processing import load_hourly, extract_experiment_period


class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        self.hourly_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "vic_elec_hourly.csv"))

    def test_hourly_dataset_exists(self):
        self.assertTrue(os.path.exists(self.hourly_path), "vic_elec_hourly.csv must exist in data/")

    def test_3601_observation_slice(self):
        """Must match paper's exact 3,601 hourly observations."""
        hourly = load_hourly(self.hourly_path)
        series = extract_experiment_period(hourly, n_obs=3601)
        self.assertEqual(len(series), 3601)
        self.assertEqual(str(series.index[0]), "2014-01-01 00:00:00")
        self.assertEqual(str(series.index[-1]), "2014-05-31 00:00:00")

    def test_demand_values_positive(self):
        """Electricity demand in MWh must be strictly positive and realistic."""
        hourly = load_hourly(self.hourly_path)
        series = extract_experiment_period(hourly, n_obs=3601)
        self.assertTrue((series > 0).all(), "Demand must be strictly positive")
        self.assertGreater(series.min(), 5000)
        self.assertLess(series.max(), 30000)


if __name__ == '__main__':
    unittest.main()

