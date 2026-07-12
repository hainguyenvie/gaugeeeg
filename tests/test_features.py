import unittest

import numpy as np

from gaugeeeg.features import BandpowerEncoder


class BandpowerTests(unittest.TestCase):
    def test_alpha_tone_has_larger_alpha_than_theta(self):
        sfreq = 200.0
        time = np.arange(800) / sfreq
        signal = np.sin(2 * np.pi * 10.0 * time)
        x = np.tile(signal, (4, 2, 1)).astype(np.float32)
        encoder = BandpowerEncoder()
        features = encoder.transform(x, ("C3", "C4"), sfreq).reshape(4, 2, 3)
        self.assertTrue(np.all(features[..., 1] > features[..., 0]))

    def test_feature_shape(self):
        x = np.random.default_rng(1).normal(size=(7, 5, 400)).astype(np.float32)
        features = BandpowerEncoder().transform(x, [f"C{i}" for i in range(5)], 200.0)
        self.assertEqual(features.shape, (7, 15))


if __name__ == "__main__":
    unittest.main()
