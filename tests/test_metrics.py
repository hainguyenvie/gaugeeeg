import unittest

import numpy as np

from gaugeeeg.metrics import linear_cka, paired_cosine, relative_l2_drift


class RepresentationMetricTests(unittest.TestCase):
    def test_identity_metrics(self):
        x = np.random.default_rng(2).normal(size=(30, 12))
        self.assertAlmostEqual(paired_cosine(x, x), 1.0, places=10)
        self.assertAlmostEqual(relative_l2_drift(x, x), 0.0, places=10)
        self.assertAlmostEqual(linear_cka(x, x), 1.0, places=10)

    def test_cka_is_invariant_to_isotropic_scaling(self):
        x = np.random.default_rng(3).normal(size=(25, 9))
        self.assertAlmostEqual(linear_cka(x, 4.0 * x), 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
