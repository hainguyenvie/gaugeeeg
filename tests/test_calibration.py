import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from gaugeeeg.calibration import analyze_calibration_controls, fit_calibrator
from gaugeeeg.cli import build_parser


class CalibrationTests(unittest.TestCase):
    def test_temperature_is_argmax_invariant_and_bias_can_recover_class_shift(self):
        labels = np.tile(np.arange(4), 40)
        logits = np.eye(4)[labels] * 1.2
        logits[:, 0] += 1.5

        temperature = fit_calibrator(logits, labels, "temperature")
        self.assertTrue(
            np.array_equal(logits.argmax(axis=1), temperature.transform(logits).argmax(axis=1))
        )

        before = balanced_accuracy_score(labels, logits.argmax(axis=1))
        bias = fit_calibrator(logits, labels, "bias")
        after = balanced_accuracy_score(labels, bias.transform(logits).argmax(axis=1))
        self.assertGreater(after, before)

    def test_end_to_end_audit_keeps_validation_and_test_subjects_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validation_path = root / "validation.csv"
            test_path = root / "test.csv"
            baseline_path = root / "baseline.csv"
            views = ("native16@car", "native16@cz", "native16@pz", "native16@fz")
            offsets = {
                "native16@car": np.array([0.7, 0.0, 0.0, 0.0]),
                "native16@cz": np.array([0.5, 0.1, 0.0, 0.0]),
                "native16@pz": np.array([0.0, 0.0, 0.0, 0.8]),
                "native16@fz": np.array([0.8, 0.0, 0.0, 0.0]),
            }
            validation = self._prediction_frame(views, offsets, subjects=np.arange(71, 75))
            test = self._prediction_frame(views, offsets, subjects=np.arange(90, 94))
            validation.to_csv(validation_path, index=False)
            test.to_csv(test_path, index=False)
            test.drop(columns=[column for column in test if column.startswith("logit_")]).to_csv(
                baseline_path,
                index=False,
            )

            output = root / "audit"
            metrics = analyze_calibration_controls(
                validation_path,
                test_path,
                output,
                baseline_predictions=baseline_path,
                n_resamples=20,
                bootstrap_seed=13,
            )
            summary = json.loads((output / "calibration_summary.json").read_text())
            self.assertTrue(summary["validation_test_subjects_disjoint"])
            self.assertFalse(summary["test_labels_used_for_fitting"])
            self.assertTrue(summary["e7e_predictions_reproduced_exactly"])
            self.assertIn("view_specific", set(metrics["protocol"]))
            self.assertIn("leave_one_view_out", set(metrics["protocol"]))
            self.assertTrue((output / "calibration_stability.csv").exists())

    def test_cli_parser_exposes_calibration_control(self):
        args = build_parser().parse_args(
            [
                "calibration-control",
                "--validation-predictions",
                "validation.csv",
                "--test-predictions",
                "test.csv",
            ]
        )
        self.assertEqual(args.command, "calibration-control")

    @staticmethod
    def _prediction_frame(
        views: tuple[str, ...],
        offsets: dict[str, np.ndarray],
        *,
        subjects: np.ndarray,
    ) -> pd.DataFrame:
        frames = []
        class_names = ("left_fist", "right_fist", "both_fists", "both_feet")
        labels = np.tile(np.arange(4), subjects.size * 3)
        subject_ids = np.repeat(subjects, 12)
        trial_index = np.arange(labels.size)
        base = np.eye(4)[labels] * 0.9
        noise = np.sin(np.arange(labels.size)[:, None] + np.arange(4)[None, :]) * 0.05
        for view in views:
            logits = base + noise + offsets[view]
            prediction = logits.argmax(axis=1)
            frame = pd.DataFrame(
                {
                    "test_view": view,
                    "trial_index": trial_index,
                    "subject_id": subject_ids,
                    "y_true": labels,
                    "y_pred": prediction,
                }
            )
            for class_index, class_name in enumerate(class_names):
                frame[f"logit_{class_name}"] = logits[:, class_index]
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    unittest.main()
