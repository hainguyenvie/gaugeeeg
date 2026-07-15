import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gaugeeeg.bias_manifold import (
    analyze_bias_manifold,
    nominal_topology_coordinate,
    topology_descriptor,
)
from gaugeeeg.cli import build_parser


class BiasManifoldTests(unittest.TestCase):
    def test_nominal_topology_preserves_laterality_and_anterior_order(self):
        c3 = nominal_topology_coordinate("C3")
        c4 = nominal_topology_coordinate("C4")
        fz = nominal_topology_coordinate("Fz")
        pz = nominal_topology_coordinate("Pz")
        self.assertLess(c3[0], 0.0)
        self.assertGreater(c4[0], 0.0)
        self.assertAlmostEqual(abs(c3[0]), abs(c4[0]))
        self.assertGreater(fz[1], pz[1])
        descriptor, names = topology_descriptor("native16@C3")
        self.assertEqual(descriptor.size, len(names))
        self.assertTrue(np.isfinite(descriptor).all())

    def test_validation_only_leave_one_electrode_out_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "validation_predictions.csv"
            frame = self._make_predictions()
            frame.to_csv(predictions, index=False)
            e8_predictions = root / "e8_validation_predictions.csv"
            e8_frame = frame.copy()
            e8_frame["test_view"] = e8_frame["test_view"].str.casefold()
            e8_frame.to_csv(e8_predictions, index=False)

            output = root / "audit"
            aggregate = analyze_bias_manifold(
                predictions,
                output,
                fit_subjects=[71, 72],
                evaluation_subjects=[81, 82],
                e8_validation_predictions=e8_predictions,
                ridge_alpha=1.0,
            )
            summary = json.loads((output / "bias_manifold_summary.json").read_text())
            self.assertTrue(summary["fit_evaluation_subjects_disjoint"])
            self.assertFalse(summary["physionet_test_subjects_used_by_manifold_analysis"])
            self.assertFalse(summary["held_out_target_labels_used_for_candidate_fitting"])
            self.assertEqual(summary["held_out_unit"], "reference electrode identity across all montages")
            self.assertTrue(
                summary["e8_shared_output_reproduction"][
                    "shared_predictions_reproduced_exactly"
                ]
            )
            self.assertEqual(
                len(summary["e8_shared_output_reproduction"]["shared_views"]),
                10,
            )
            self.assertEqual(
                set(aggregate["method"]),
                {
                    "identity",
                    "global_mean",
                    "pooled_bias",
                    "topology_ridge",
                    "logit_ridge",
                    "combined_ridge",
                    "oracle",
                },
            )
            metrics = pd.read_csv(output / "bias_manifold_metrics.csv")
            deployable = metrics.loc[metrics["method"] != "oracle"]
            self.assertFalse(deployable["target_reference_labels_used"].any())

    def test_cli_parser_exposes_bias_manifold(self):
        args = build_parser().parse_args(
            [
                "bias-manifold",
                "--validation-predictions",
                "validation.csv",
            ]
        )
        self.assertEqual(args.fit_subjects, list(range(71, 81)))
        self.assertEqual(args.evaluation_subjects, list(range(81, 90)))

    def test_target_fit_labels_cannot_change_held_out_bias_prediction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = self._make_predictions()
            changed = original.copy()
            target_fit = (
                changed["test_view"].str.casefold().str.endswith("@c3")
                & changed["subject_id"].isin([71, 72])
            )
            changed.loc[target_fit, "y_true"] = (changed.loc[target_fit, "y_true"] + 1) % 4
            paths = []
            for name, frame in (("original", original), ("changed", changed)):
                path = root / f"{name}.csv"
                frame.to_csv(path, index=False)
                analyze_bias_manifold(
                    path,
                    root / name,
                    fit_subjects=[71, 72],
                    evaluation_subjects=[81, 82],
                )
                paths.append(root / name / "bias_manifold_predictions.csv")
            first, second = (pd.read_csv(path) for path in paths)
            methods = [
                "global_mean",
                "pooled_bias",
                "topology_ridge",
                "logit_ridge",
                "combined_ridge",
            ]
            selector = lambda frame: frame.loc[
                (frame["held_out_reference"] == "c3") & frame["method"].isin(methods),
                ["target_view", "method", "parameter_index", "predicted_bias"],
            ].sort_values(["target_view", "method", "parameter_index"]).reset_index(drop=True)
            pd.testing.assert_frame_equal(selector(first), selector(second))

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
        frames = []
        class_names = ("left_fist", "right_fist", "both_fists", "both_feet")
        for view in views:
            descriptor, _ = topology_descriptor(view)
            x, y, montage_size = descriptor[0], descriptor[1], descriptor[12]
            offset = np.asarray(
                [1.1 * x + 0.2 * montage_size, 0.8 * y, -0.7 * x, -0.5 * y]
            )
            base = np.eye(4)[labels] * 1.0
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
