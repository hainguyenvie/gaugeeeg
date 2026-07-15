import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gaugeeeg.bias_manifold import topology_descriptor
from gaugeeeg.calibration import fit_calibrator
from gaugeeeg.cli import build_parser
from gaugeeeg.prior_stress import analyze_prior_stress, fit_known_prior_bias


class PriorStressTests(unittest.TestCase):
    def test_known_prior_objective_matches_supervised_bias(self):
        rng = np.random.default_rng(7)
        labels = np.tile(np.arange(4), 25)
        logits = rng.normal(size=(labels.size, 4)) + np.eye(4)[labels]
        empirical_prior = np.bincount(labels, minlength=4) / labels.size
        supervised = fit_calibrator(logits, labels, "bias")
        label_free = fit_known_prior_bias(logits, empirical_prior)
        self.assertTrue(label_free.success, label_free.message)
        np.testing.assert_allclose(
            label_free.parameters,
            supervised.parameters,
            atol=5e-5,
            rtol=0.0,
        )

    def test_cpu_only_prior_stress_writes_predeclared_suite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "validation_predictions.csv"
            self._make_predictions().to_csv(predictions, index=False)
            output = root / "audit"
            aggregate = analyze_prior_stress(
                predictions,
                output,
                fit_subjects=[71, 72],
                evaluation_subjects=[81, 82],
                batch_sizes=[4, 8, 16],
                primary_batch_size=4,
                stress_batch_size=4,
                n_resamples=2,
                bootstrap_resamples=50,
            )
            summary = json.loads((output / "prior_stress_summary.json").read_text())
            self.assertTrue(summary["fit_evaluation_subjects_disjoint"])
            self.assertFalse(summary["physionet_test_subjects_used"])
            self.assertFalse(summary["held_out_target_labels_used_for_candidate_fitting"])
            self.assertLess(
                summary["empirical_prior_match_mean_rmse_to_supervised_oracle"],
                5e-5,
            )
            self.assertIn("random", set(aggregate["condition"]))
            self.assertIn("balanced", set(aggregate["condition"]))
            self.assertIn("skew_0.7_class_0", set(aggregate["condition"]))
            self.assertEqual(set(aggregate["method"]), {
                "identity",
                "topology_ridge",
                "prior_match",
                "topology_shrinkage",
                "oracle",
            })
            for filename in (
                "prior_stress_metrics.csv",
                "prior_stress_aggregate.csv",
                "prior_stress_weights.csv",
                "prior_stress_bias_audit.csv",
                "prior_stress_selections.csv",
                "prior_stress_bootstrap.csv",
            ):
                self.assertTrue((output / filename).is_file(), filename)

    def test_held_out_labels_cannot_change_deployable_target_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = self._make_predictions()
            changed = original.copy()
            candidates = changed.loc[
                changed["test_view"].str.casefold().str.endswith("@c3")
                & changed["subject_id"].isin([71, 72])
                & changed["y_true"].eq(3)
            ]
            indices = candidates.groupby("test_view", sort=False).head(1).index
            changed.loc[indices, "y_true"] = 0

            outputs = []
            for name, frame in (("original", original), ("changed", changed)):
                path = root / f"{name}.csv"
                frame.to_csv(path, index=False)
                output = root / name
                analyze_prior_stress(
                    path,
                    output,
                    fit_subjects=[71, 72],
                    evaluation_subjects=[81, 82],
                    batch_sizes=[4, 8, 16],
                    primary_batch_size=4,
                    stress_batch_size=4,
                    n_resamples=2,
                    bootstrap_resamples=50,
                )
                outputs.append(output)

            methods = ["topology_ridge", "prior_match", "topology_shrinkage"]
            metric_columns = [
                "condition",
                "batch_size",
                "repeat",
                "target_view",
                "method",
                "prior_match_weight",
                "balanced_accuracy",
                "nll",
                "ece_15",
                "max_class_recall_gap_to_car",
                "mean_class_recall_gap_to_car",
            ]

            def deployable_metrics(output: Path) -> pd.DataFrame:
                frame = pd.read_csv(output / "prior_stress_metrics.csv")
                return frame.loc[
                    frame["held_out_reference"].eq("c3")
                    & frame["method"].isin(methods),
                    metric_columns,
                ].reset_index(drop=True)

            pd.testing.assert_frame_equal(
                deployable_metrics(outputs[0]),
                deployable_metrics(outputs[1]),
            )
            first_weights = pd.read_csv(outputs[0] / "prior_stress_weights.csv")
            second_weights = pd.read_csv(outputs[1] / "prior_stress_weights.csv")
            selector = lambda frame: frame.loc[
                frame["held_out_reference"].eq("c3")
            ].reset_index(drop=True)
            pd.testing.assert_frame_equal(selector(first_weights), selector(second_weights))

    def test_cli_parser_exposes_prior_stress_defaults(self):
        args = build_parser().parse_args(
            ["prior-stress", "--validation-predictions", "validation.csv"]
        )
        self.assertEqual(args.primary_batch_size, 32)
        self.assertEqual(args.stress_batch_size, 128)
        self.assertEqual(args.batch_sizes[-1], 900)

    @staticmethod
    def _make_predictions() -> pd.DataFrame:
        views = [
            f"{montage}@{reference}"
            for montage in ("native16", "native32")
            for reference in ("car", "C3", "C4", "Cz", "Pz")
        ]
        subjects = np.asarray([71, 72, 81, 82])
        labels = np.tile(np.arange(4), subjects.size * 2)
        subject_ids = np.repeat(subjects, 8)
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
