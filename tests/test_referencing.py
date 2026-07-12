import unittest

import numpy as np

from gaugeeeg.referencing import (
    apply_reference_view,
    common_average,
    pairwise_difference,
    reference_matrix,
    single_reference,
    weighted_reference,
)


class ReferencingTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12)
        self.x = self.rng.normal(size=(5, 6, 100)).astype(np.float32)
        self.names = ("Fp1", "Fz", "Cz", "Pz", "O1", "O2")

    def test_car_removes_common_signal(self):
        common = self.rng.normal(size=(5, 1, 100)).astype(np.float32)
        np.testing.assert_allclose(common_average(self.x), common_average(self.x + common), atol=1e-6)

    def test_car_recovers_after_single_reference(self):
        car = common_average(self.x)
        shifted = single_reference(car, 2)
        np.testing.assert_allclose(common_average(shifted), car, atol=1e-6)

    def test_car_recovers_after_weighted_reference(self):
        car = common_average(self.x)
        weights = self.rng.dirichlet(np.ones(car.shape[-2]))
        shifted = weighted_reference(car, weights)
        np.testing.assert_allclose(common_average(shifted), car, atol=1e-6)

    def test_pairwise_differences_are_invariant(self):
        shifted = single_reference(self.x, 4)
        np.testing.assert_allclose(
            pairwise_difference(self.x, 1, 5),
            pairwise_difference(shifted, 1, 5),
            atol=1e-6,
        )

    def test_reference_matrix_annihilates_ones(self):
        weights = self.rng.dirichlet(np.ones(6))
        matrix = reference_matrix(weights)
        np.testing.assert_allclose(matrix @ np.ones(6), 0.0, atol=1e-12)

    def test_named_views_are_deterministic(self):
        first = apply_reference_view(self.x, self.names, "random_linear", seed=7)
        second = apply_reference_view(self.x, self.names, "random_linear", seed=7)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
