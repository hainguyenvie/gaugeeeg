import unittest

import numpy as np

from gaugeeeg.montage import (
    SPARSE_MONTAGES,
    apply_observation_view,
    montage_keep_mask,
    observation_metadata,
    prepare_observation_view,
    zero_fill_unobserved,
)
from gaugeeeg.referencing import common_average


class MontageTests(unittest.TestCase):
    def setUp(self):
        self.names = tuple(dict.fromkeys(SPARSE_MONTAGES["sparse32"] + ("Oz", "Fp1")))
        self.rng = np.random.default_rng(14)
        self.x = self.rng.normal(size=(4, len(self.names), 80)).astype(np.float32)

    def test_sparse_montages_are_nested_and_have_declared_sizes(self):
        masks = {name: montage_keep_mask(self.names, name) for name in SPARSE_MONTAGES}
        self.assertEqual(int(masks["sparse8"].sum()), 8)
        self.assertEqual(int(masks["sparse16"].sum()), 16)
        self.assertEqual(int(masks["sparse32"].sum()), 32)
        self.assertTrue(np.all(masks["sparse8"] <= masks["sparse16"]))
        self.assertTrue(np.all(masks["sparse16"] <= masks["sparse32"]))

    def test_zero_fill_preserves_shape_and_observed_values(self):
        mask = montage_keep_mask(self.names, "sparse8")
        masked = zero_fill_unobserved(self.x, self.names, "sparse8")
        self.assertEqual(masked.shape, self.x.shape)
        np.testing.assert_array_equal(masked[:, mask], self.x[:, mask])
        np.testing.assert_array_equal(masked[:, ~mask], 0.0)

    def test_composite_view_references_before_masking(self):
        composite = apply_observation_view(self.x, self.names, "sparse16@cz")
        referenced = apply_observation_view(self.x, self.names, "cz")
        expected = zero_fill_unobserved(referenced, self.names, "sparse16")
        np.testing.assert_array_equal(composite, expected)

    def test_car_cannot_recover_channels_removed_by_sparse_observation(self):
        clean = common_average(self.x)
        sparse = apply_observation_view(clean, self.names, "sparse8@cz")
        recovered = common_average(sparse)
        self.assertGreater(float(np.max(np.abs(recovered - clean))), 1e-3)

    def test_metadata_records_explicit_zero_fill(self):
        metadata = observation_metadata(self.names, "sparse16@car")
        self.assertEqual(metadata["reference_view"], "car")
        self.assertEqual(metadata["n_observed_channels"], 16)
        self.assertEqual(metadata["missing_channel_fill"], "zero")

    def test_native_view_removes_channels_and_aligned_names(self):
        observed, names = prepare_observation_view(self.x, self.names, "native16@cz")
        mask = montage_keep_mask(self.names, "sparse16")
        subset = self.x[..., mask, :]
        self.assertEqual(observed.shape[-2], 16)
        self.assertEqual(names, tuple(name for name, keep in zip(self.names, mask, strict=True) if keep))
        expected = apply_observation_view(subset, names, "cz")
        np.testing.assert_array_equal(observed, expected)
        metadata = observation_metadata(self.names, "native16@cz")
        self.assertEqual(metadata["missing_channel_fill"], "removed")
        self.assertEqual(metadata["channel_policy"], "remove")

    def test_native_car_is_computed_within_retained_montage(self):
        native_car, names = prepare_observation_view(self.x, self.names, "native16@car")
        native_cz, cz_names = prepare_observation_view(self.x, self.names, "native16@cz")
        self.assertEqual(names, cz_names)
        np.testing.assert_allclose(native_car.mean(axis=-2), 0.0, atol=1e-6)
        np.testing.assert_allclose(common_average(native_cz), native_car, atol=1e-6)

    def test_unknown_montage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown montage"):
            apply_observation_view(self.x, self.names, "unknown@car")


if __name__ == "__main__":
    unittest.main()
