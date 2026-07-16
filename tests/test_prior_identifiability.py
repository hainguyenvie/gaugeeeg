import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gaugeeeg.bias_manifold import topology_descriptor
from gaugeeeg.cli import build_parser
from gaugeeeg.prior_identifiability import (
    analyze_prior_identifiability,
    estimate_regularized_soft_prior,
)


class PriorIdentifiabilityTests(unittest.TestCase):
    def test_regularized_soft_prior_is_simplex_and_shrinks_to_nominal(self):
        confusion = np.eye(4)
        observed = np.asarray([0.7, 0.1, 0.1, 0.1])
        nominal = np.full(4, 0.25)
        unregularized = estimate_regularized_soft_prior(
            confusion,
            observed,
            nominal,
            regularization=0.0,
        )
        regularized = estimate_regularized_soft_prior(
            confusion,
            observed,
            nominal,
            regularization=1.0,
        )
        np.testing.assert_allclose(unregularized, observed)
        self.assertAlmostEqual(float(regularized.sum()), 1.0)
        self.assertTrue((regularized >= 0.0).all())
        self.assertLess(
            np.linalg.norm(regularized - nominal),
            np.linalg.norm(unregularized - nominal),
        )

    def test_cross_subject_audit_writes_complete_output_suite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "validation_predictions.csv"
            self._make_predictions().to_csv(predictions, index=False)
            output = root / "audit"
            aggregate = analyze_prior_identifiability(
                predictions,
                output,
                topology_subjects=[71, 72],
                prior_model_subjects=[71, 72],
                adaptation_subjects=[73, 74],
                evaluation_subjects=[81, 82],
                batch_sizes=[4, 8, 16, 32],
                primary_batch_size=4,
                stress_batch_size=8,
                n_resamples=2,
                bootstrap_resamples=50,
            )
            summary = json.loads(
                (output / "prior_identifiability_summary.json").read_text()
            )
            self.assertTrue(summary["prior_model_adaptation_subjects_disjoint"])
            self.assertTrue(summary["topology_adaptation_subjects_disjoint"])
            self.assertTrue(summary["topology_evaluation_subjects_disjoint"])
            self.assertTrue(summary["adaptation_evaluation_subjects_disjoint"])
            self.assertTrue(summary["source_and_target_batch_seeds_disjoint"])
            self.assertFalse(summary["physionet_test_subjects_used"])
            self.assertFalse(summary["target_reference_labels_used_for_candidate"])
            self.assertTrue(
                summary["soft_confusion_uses_leave_one_subject_out_predictions"]
            )
            self.assertIn("mean_severe_robustness_supported", summary)
            self.assertIn("class_uniform_severe_robustness_supported", summary)
            self.assertIn("n_severe_dominant_classes_improved", summary)
            self.assertEqual(
                summary["strict_severe_skew_improved"],
                summary["severe_skew_improved"],
            )
            self.assertEqual(
                set(aggregate["method"]),
                {
                    "identity",
                    "topology_ridge",
                    "uniform_prior_match",
                    "fixed_topology_shrinkage",
                    "operator_confusion_shrinkage",
                    "oracle",
                },
            )
            for filename in (
                "prior_identifiability_metrics.csv",
                "prior_identifiability_aggregate.csv",
                "prior_identifiability_estimates.csv",
                "prior_identifiability_ablation.csv",
                "prior_identifiability_weights.csv",
                "prior_identifiability_models.csv",
                "prior_identifiability_selections.csv",
                "prior_identifiability_bootstrap.csv",
            ):
                self.assertTrue((output / filename).is_file(), filename)

    def test_held_out_reference_labels_cannot_change_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = self._make_predictions()
            changed = original.copy()
            candidates = changed.loc[
                changed["test_view"].str.casefold().str.endswith("@c3")
                & changed["subject_id"].isin([73, 74, 81, 82])
                & changed["y_true"].eq(3)
            ]
            indices = candidates.groupby("test_view", sort=False).head(1).index
            changed.loc[indices, "y_true"] = 0

            outputs = []
            for name, frame in (("original", original), ("changed", changed)):
                path = root / f"{name}.csv"
                frame.to_csv(path, index=False)
                output = root / name
                analyze_prior_identifiability(
                    path,
                    output,
                    topology_subjects=[71, 72],
                    prior_model_subjects=[71, 72],
                    adaptation_subjects=[73, 74],
                    evaluation_subjects=[81, 82],
                    batch_sizes=[4, 8, 16, 32],
                    primary_batch_size=4,
                    stress_batch_size=8,
                    n_resamples=2,
                    bootstrap_resamples=50,
                )
                outputs.append(output)

            columns = [
                "condition",
                "batch_size",
                "repeat",
                "target_view",
                "soft_mean_prior",
                "weak_confusion_prior",
                "regularized_confusion_prior",
                "prior_match_weight",
                "candidate_bias",
                "target_reference_labels_used_for_candidate",
            ]

            def candidate_rows(output: Path) -> pd.DataFrame:
                frame = pd.read_csv(output / "prior_identifiability_estimates.csv")
                return frame.loc[
                    frame["held_out_reference"].eq("c3"), columns
                ].reset_index(drop=True)

            pd.testing.assert_frame_equal(
                candidate_rows(outputs[0]),
                candidate_rows(outputs[1]),
            )

    def test_cli_parser_exposes_prior_identifiability_defaults(self):
        args = build_parser().parse_args(
            [
                "prior-identifiability",
                "--validation-predictions",
                "validation.csv",
            ]
        )
        self.assertEqual(args.topology_subjects, [71, 72, 73, 74, 75])
        self.assertEqual(args.prior_model_subjects, [71, 72, 73, 74, 75])
        self.assertEqual(args.adaptation_subjects, [76, 77, 78, 79, 80])
        self.assertEqual(args.batch_sizes[-1], 450)
        self.assertEqual(args.confusion_regularization, 1.0)

    @staticmethod
    def _make_predictions() -> pd.DataFrame:
        views = [
            f"{montage}@{reference}"
            for montage in ("native16", "native32")
            for reference in ("car", "C3", "C4", "Cz", "Pz")
        ]
        subjects = np.asarray([71, 72, 73, 74, 81, 82])
        labels = np.tile(np.repeat(np.arange(4), 4), subjects.size)
        subject_ids = np.repeat(subjects, 16)
        trial_index = np.arange(labels.size)
        class_names = ("left_fist", "right_fist", "both_fists", "both_feet")
        frames = []
        for view in views:
            descriptor, _ = topology_descriptor(view)
            x, y, montage_size = descriptor[0], descriptor[1], descriptor[12]
            offset = np.asarray(
                [1.1 * x + 0.2 * montage_size, 0.8 * y, -0.7 * x, -0.5 * y]
            )
            base = np.eye(4)[labels]
            noise = np.sin(trial_index[:, None] + np.arange(4)[None, :]) * 0.04
            logits = base + offset + noise
            frame = pd.DataFrame(
                {
                    "test_view": view,
                    "trial_index": trial_index,
                    "subject_id": subject_ids,
                    "y_true": labels,
                    "y_pred": logits.argmax(axis=1),
                }
            )
            for class_index, class_name in enumerate(class_names):
                frame[f"logit_{class_name}"] = logits[:, class_index]
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    unittest.main()
