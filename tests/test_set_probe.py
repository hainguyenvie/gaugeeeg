import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
class ReveSetProbeTests(unittest.TestCase):
    def test_predictor_accepts_fewer_tokens_than_training(self):
        from gaugeeeg.set_probe import fit_reve_set_probe

        rng = np.random.default_rng(7)
        train_x = rng.normal(size=(48, 7, 8)).astype(np.float32)
        train_y = np.tile(np.arange(4), 12)
        val_x = rng.normal(size=(16, 7, 8)).astype(np.float32)
        val_y = np.tile(np.arange(4), 4)
        result = fit_reve_set_probe(
            train_x,
            train_y,
            val_x,
            val_y,
            initial_query=np.zeros(8, dtype=np.float32),
            n_classes=4,
            seed=7,
            device="cpu",
            n_queries=2,
            n_heads=2,
            ff_multiplier=1,
            batch_size=8,
            epochs=2,
            warmup_epochs=0,
            patience=1,
        )
        shorter = rng.normal(size=(5, 3, 8)).astype(np.float32)
        self.assertEqual(result.model.predict(shorter).shape, (5,))
        self.assertEqual(result.model.predict_proba(shorter).shape, (5, 4))

    def test_rejects_non_token_features(self):
        from gaugeeeg.set_probe import fit_reve_set_probe

        with self.assertRaisesRegex(ValueError, "shape"):
            fit_reve_set_probe(
                np.zeros((8, 4), dtype=np.float32),
                np.tile(np.arange(4), 2),
                np.zeros((4, 4), dtype=np.float32),
                np.arange(4),
                initial_query=np.zeros(4, dtype=np.float32),
                n_classes=4,
                seed=7,
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
