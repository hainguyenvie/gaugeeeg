import unittest

import numpy as np


class TorchProbeTests(unittest.TestCase):
    def test_token_probe_learns_separable_toy_data(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed")

        from gaugeeeg.torch_probe import fit_reve_token_probe

        rng = np.random.default_rng(4)
        x = rng.normal(scale=0.1, size=(80, 3, 4)).astype(np.float32)
        y = np.tile(np.arange(4), 20)
        x[np.arange(80), 0, y] += 3.0
        result = fit_reve_token_probe(
            x[:64],
            y[:64],
            x[64:],
            y[64:],
            initial_query=np.zeros(4, dtype=np.float32),
            n_classes=4,
            seed=2,
            device="cpu",
            batch_size=16,
            epochs=8,
            learning_rate=0.03,
            warmup_epochs=0,
            patience=8,
        )
        self.assertGreaterEqual(result.validation_balanced_accuracy, 0.9)
        self.assertGreaterEqual(len(result.history), 1)
        self.assertIn("learning_rate", result.history[0])

    def test_multi_view_consistency_probe_accepts_aligned_views(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed")

        from gaugeeeg.torch_probe import fit_reve_token_probe

        rng = np.random.default_rng(8)
        clean = rng.normal(scale=0.1, size=(80, 3, 4)).astype(np.float32)
        labels = np.tile(np.arange(4), 20)
        clean[np.arange(80), 0, labels] += 3.0
        shifted = clean + rng.normal(scale=0.02, size=clean.shape).astype(np.float32)
        views = np.stack([clean, shifted], axis=1)
        result = fit_reve_token_probe(
            views[:64],
            labels[:64],
            views[64:],
            labels[64:],
            initial_query=np.zeros(4, dtype=np.float32),
            n_classes=4,
            seed=3,
            device="cpu",
            batch_size=16,
            epochs=8,
            learning_rate=0.03,
            warmup_epochs=0,
            patience=8,
            objective="rule_consistency",
            consistency_weight=1.0,
        )
        self.assertGreaterEqual(result.validation_balanced_accuracy, 0.9)
        self.assertIn("train_consistency_loss", result.history[0])


if __name__ == "__main__":
    unittest.main()
