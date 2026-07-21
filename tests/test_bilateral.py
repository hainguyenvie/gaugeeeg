import unittest

import numpy as np

from gaugeeeg.bilateral import BILATERAL_MOTOR_PAIRS, bilateral_spectral_tokens


class BilateralSpectralTokenTests(unittest.TestCase):
    def setUp(self):
        names = []
        for left, right, midlines in BILATERAL_MOTOR_PAIRS:
            names.extend([left, right, *midlines])
        self.channel_names = tuple(dict.fromkeys(names))
        rng = np.random.default_rng(7)
        self.signal = rng.normal(size=(5, len(self.channel_names), 320)).astype(np.float32)
        self.reference = rng.normal(size=(5, 1, 320)).astype(np.float32)

    def test_odd_even_is_exactly_reference_invariant_to_numerical_tolerance(self):
        original = bilateral_spectral_tokens(
            self.signal,
            self.channel_names,
            160.0,
            mode="gqba_odd_even",
        )
        shifted = bilateral_spectral_tokens(
            self.signal - self.reference,
            self.channel_names,
            160.0,
            mode="gqba_odd_even",
        )
        np.testing.assert_allclose(original, shifted, rtol=2e-5, atol=2e-5)
        self.assertEqual(original.shape, (5, 6, 9))

    def test_capacity_control_is_not_reference_invariant(self):
        original = bilateral_spectral_tokens(
            self.signal,
            self.channel_names,
            160.0,
            mode="spectral_capacity_control",
        )
        shifted = bilateral_spectral_tokens(
            self.signal - 5.0 * self.reference,
            self.channel_names,
            160.0,
            mode="spectral_capacity_control",
        )
        self.assertGreater(float(np.max(np.abs(original - shifted))), 1e-3)

    def test_odd_only_reserves_even_and_control_slots(self):
        tokens = bilateral_spectral_tokens(
            self.signal,
            self.channel_names,
            160.0,
            mode="gqba_odd",
        )
        self.assertTrue(np.any(tokens[..., :3] != 0.0))
        self.assertTrue(np.all(tokens[..., 3:] == 0.0))

    def test_missing_core_channel_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unavailable motor channels"):
            bilateral_spectral_tokens(
                self.signal[:, 1:],
                self.channel_names[1:],
                160.0,
                mode="gqba_odd_even",
            )


if __name__ == "__main__":
    unittest.main()
