import unittest

import numpy as np

from gaugeeeg.metrics import (
    linear_cka,
    paired_cosine,
    paired_subject_bootstrap_bacc_gap,
    relative_l2_drift,
)


class RepresentationMetricTests(unittest.TestCase):
    def test_identity_metrics(self):
        x = np.random.default_rng(2).normal(size=(30, 12))
        self.assertAlmostEqual(paired_cosine(x, x), 1.0, places=10)
        self.assertAlmostEqual(relative_l2_drift(x, x), 0.0, places=10)
        self.assertAlmostEqual(linear_cka(x, x), 1.0, places=10)

    def test_cka_is_invariant_to_isotropic_scaling(self):
        x = np.random.default_rng(3).normal(size=(25, 9))
        self.assertAlmostEqual(linear_cka(x, 4.0 * x), 1.0, places=10)


class SubjectBootstrapTests(unittest.TestCase):
    def test_identity_has_zero_gap_and_zero_width_interval(self):
        y = np.tile(np.arange(4), 6)
        subjects = np.repeat(np.arange(3), 8)
        result = paired_subject_bootstrap_bacc_gap(
            y,
            y,
            y,
            subjects,
            n_resamples=100,
            seed=5,
        )
        self.assertEqual(result["point_gap"], 0.0)
        self.assertEqual(result["ci_lower"], 0.0)
        self.assertEqual(result["ci_upper"], 0.0)

    def test_consistent_subject_level_failure_has_positive_interval(self):
        y = np.tile(np.arange(4), 10)
        subjects = np.repeat(np.arange(5), 8)
        shifted = np.zeros_like(y)
        result = paired_subject_bootstrap_bacc_gap(
            y,
            y,
            shifted,
            subjects,
            n_resamples=100,
            seed=9,
        )
        self.assertGreater(result["point_gap"], 0.0)
        self.assertGreater(result["ci_lower"], 0.0)


if __name__ == "__main__":
    unittest.main()
