import unittest

import numpy as np

from gaugeeeg.channel_adaptation import (
    adapt_observation_channels,
    spherical_spline_interpolation_matrix,
    spherical_spline_to_montage,
)


class ChannelAdaptationTests(unittest.TestCase):
    def setUp(self):
        self.positions = {
            "a": np.array([1.0, 0.0, 0.0]),
            "b": np.array([0.0, 1.0, 0.0]),
            "c": np.array([0.0, 0.0, 1.0]),
            "d": np.array([-1.0, -1.0, -1.0]),
            "e": np.array([1.0, 1.0, 1.0]),
        }

    def test_spherical_spline_weights_preserve_constant_field(self):
        observed = np.stack([self.positions[name] for name in ("a", "b", "c", "d")])
        target = np.stack([self.positions["e"]])
        weights = spherical_spline_interpolation_matrix(observed, target)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-9)

    def test_interpolation_preserves_measured_channels_and_order(self):
        signal = np.array(
            [
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0],
                ]
            ],
            dtype=np.float32,
        )
        output = spherical_spline_to_montage(
            signal,
            ("a", "b", "c", "d"),
            ("e", "c", "a", "d", "b"),
            positions=self.positions,
        )
        self.assertEqual(output.shape, (1, 5, 3))
        np.testing.assert_array_equal(output[:, 1], signal[:, 2])
        np.testing.assert_array_equal(output[:, 2], signal[:, 0])
        np.testing.assert_array_equal(output[:, 3], signal[:, 3])
        np.testing.assert_array_equal(output[:, 4], signal[:, 1])

    def test_full_montage_is_identity_without_loading_mne(self):
        signal = np.arange(12, dtype=np.float32).reshape(1, 2, 6)
        output, names = adapt_observation_channels(
            signal,
            ("C3", "C4"),
            ("C3", "C4"),
            "spherical_spline",
        )
        self.assertIs(output, signal)
        self.assertEqual(names, ("C3", "C4"))

    def test_rejects_observed_channel_outside_target(self):
        with self.assertRaisesRegex(ValueError, "absent from target montage"):
            spherical_spline_to_montage(
                np.zeros((1, 2, 4), dtype=np.float32),
                ("a", "missing"),
                ("a", "b"),
                positions=self.positions,
            )


if __name__ == "__main__":
    unittest.main()
