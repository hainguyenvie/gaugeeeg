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

    def test_operator_consistency_accepts_variable_token_views(self):
        from gaugeeeg.set_probe import fit_reve_set_probe

        rng = np.random.default_rng(11)
        labels = np.tile(np.arange(4), 16)
        full = rng.normal(scale=0.1, size=(64, 8, 8)).astype(np.float32)
        native32 = full[:, :5].copy()
        native16 = full[:, :3].copy()
        for trial, label in enumerate(labels):
            full[trial, 0, label] += 3.0
            native32[trial, 0, label] += 3.0
            native16[trial, 0, label] += 3.0

        result = fit_reve_set_probe(
            (full[:48], native32[:48], native16[:48]),
            labels[:48],
            (full[48:], native32[48:], native16[48:]),
            labels[48:],
            initial_query=np.zeros(8, dtype=np.float32),
            n_classes=4,
            seed=11,
            device="cpu",
            n_queries=2,
            n_heads=2,
            ff_multiplier=1,
            batch_size=8,
            epochs=4,
            learning_rate=0.02,
            warmup_epochs=0,
            patience=4,
            objective="operator_consistency",
            consistency_weight=1.0,
            consistency_view_weights=[0.0, 0.5, 1.0],
        )
        self.assertGreaterEqual(result.validation_balanced_accuracy, 0.5)
        self.assertGreaterEqual(result.validation_consistency_loss, 0.0)

    def test_auxiliary_branch_accepts_aligned_spectral_tokens(self):
        from gaugeeeg.set_probe import fit_reve_set_probe

        rng = np.random.default_rng(17)
        labels = np.tile(np.arange(4), 12)
        tokens = rng.normal(size=(48, 5, 8)).astype(np.float32)
        auxiliary = rng.normal(size=(48, 6, 9)).astype(np.float32)
        for trial, label in enumerate(labels):
            auxiliary[trial, 0, label] += 4.0
        result = fit_reve_set_probe(
            tokens[:32],
            labels[:32],
            tokens[32:],
            labels[32:],
            initial_query=np.zeros(8, dtype=np.float32),
            n_classes=4,
            seed=17,
            device="cpu",
            n_queries=2,
            n_heads=2,
            ff_multiplier=1,
            batch_size=8,
            epochs=3,
            learning_rate=0.02,
            warmup_epochs=0,
            patience=3,
            train_auxiliary=auxiliary[:32],
            val_auxiliary=auxiliary[32:],
            auxiliary_queries=1,
            auxiliary_hidden_dim=8,
        )
        probabilities = result.model.predict_proba((tokens[32:], auxiliary[32:]))
        self.assertEqual(probabilities.shape, (16, 4))
        self.assertGreater(result.auxiliary_parameters, 0)
        self.assertGreater(result.trainable_parameters, result.auxiliary_parameters)

    def test_gated_auxiliary_reports_selective_residual_diagnostics(self):
        from gaugeeeg.set_probe import fit_reve_set_probe

        rng = np.random.default_rng(23)
        labels = np.tile(np.arange(4), 16)
        full = rng.normal(scale=0.2, size=(64, 6, 8)).astype(np.float32)
        sparse = full[:, :4].copy()
        auxiliary = rng.normal(scale=0.2, size=(64, 6, 9)).astype(np.float32)
        for trial, label in enumerate(labels):
            full[trial, 0, label] += 2.0
            sparse[trial, 0, label] += 2.0
            if label == 2:
                auxiliary[trial, 0, 0] += 3.0
        result = fit_reve_set_probe(
            (full[:48], sparse[:48]),
            labels[:48],
            (full[48:], sparse[48:]),
            labels[48:],
            initial_query=np.zeros(8, dtype=np.float32),
            n_classes=4,
            seed=23,
            device="cpu",
            n_queries=2,
            n_heads=2,
            ff_multiplier=1,
            batch_size=8,
            epochs=3,
            learning_rate=0.02,
            warmup_epochs=0,
            patience=3,
            objective="multi_view_ce",
            train_auxiliary=(auxiliary[:48], auxiliary[:48]),
            val_auxiliary=(auxiliary[48:], auxiliary[48:]),
            auxiliary_queries=1,
            auxiliary_hidden_dim=8,
            auxiliary_fusion="gated_residual",
            auxiliary_preservation_weight=1.0,
            auxiliary_residual_consistency_weight=0.1,
            auxiliary_gate_supervision_weight=0.1,
            auxiliary_target_classes=[2, 3],
        )
        diagnostics = result.model.predict_auxiliary_components((full[48:], auxiliary[48:]))
        self.assertEqual(diagnostics["gate"].shape, (16, 1))
        self.assertTrue(np.all((diagnostics["gate"] > 0.0) & (diagnostics["gate"] < 1.0)))
        self.assertTrue(np.isfinite(result.validation_auxiliary_preservation_loss))
        self.assertTrue(np.isfinite(result.validation_auxiliary_consistency_loss))
        self.assertTrue(np.isfinite(result.validation_auxiliary_gate_target_mean))
        self.assertTrue(np.isfinite(result.validation_auxiliary_gate_nontarget_mean))
        self.assertIn("train_auxiliary_preservation_loss", result.history[0])
        self.assertIn("train_auxiliary_residual_consistency_loss", result.history[0])
        self.assertIn("train_auxiliary_gate_supervision_loss", result.history[0])
        self.assertTrue(np.isfinite(result.validation_auxiliary_gate_supervision_loss))

    def test_film_auxiliary_reports_representation_diagnostics(self):
        from gaugeeeg.set_probe import fit_reve_set_probe

        rng = np.random.default_rng(29)
        labels = np.tile(np.arange(4), 16)
        full = rng.normal(scale=0.2, size=(64, 6, 8)).astype(np.float32)
        sparse = full[:, :4].copy()
        auxiliary = rng.normal(scale=0.2, size=(64, 6, 9)).astype(np.float32)
        for trial, label in enumerate(labels):
            full[trial, 0, label] += 2.0
            sparse[trial, 0, label] += 2.0
            auxiliary[trial, 0, label] += 2.0
        result = fit_reve_set_probe(
            (full[:48], sparse[:48]),
            labels[:48],
            (full[48:], sparse[48:]),
            labels[48:],
            initial_query=np.zeros(8, dtype=np.float32),
            n_classes=4,
            seed=29,
            device="cpu",
            n_queries=2,
            n_heads=2,
            ff_multiplier=1,
            batch_size=8,
            epochs=3,
            learning_rate=0.02,
            warmup_epochs=0,
            patience=3,
            objective="multi_view_ce",
            train_auxiliary=(auxiliary[:48], auxiliary[:48]),
            val_auxiliary=(auxiliary[48:], auxiliary[48:]),
            auxiliary_queries=1,
            auxiliary_hidden_dim=8,
            auxiliary_fusion="film",
            auxiliary_target_classes=[2, 3],
            representation_contrastive_weight=0.1,
            representation_bilaterality_weight=0.2,
            representation_temperature=0.1,
        )
        diagnostics = result.model.predict_representation_components((full[48:], auxiliary[48:]))
        self.assertEqual(diagnostics["representation"].shape, (16, 8))
        self.assertEqual(diagnostics["bilaterality_logits"].shape, (16, 2))
        self.assertTrue(np.isfinite(result.validation_representation_alignment_loss))
        self.assertTrue(np.isfinite(result.validation_representation_class_margin))
        self.assertTrue(np.isfinite(result.validation_representation_bilaterality_balanced_accuracy))
        self.assertIn("train_representation_contrastive_loss", result.history[0])
        self.assertIn("train_representation_bilaterality_loss", result.history[0])


if __name__ == "__main__":
    unittest.main()
