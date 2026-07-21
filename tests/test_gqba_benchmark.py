import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gaugeeeg.baseline_benchmark import EVALUATION_VIEWS, JOINT_TRAINING_VIEWS
from gaugeeeg.cli import build_parser
from gaugeeeg.gqba_benchmark import GQBA_MODES, analyze_gqba_benchmark


class GQBABenchmarkTests(unittest.TestCase):
    def _write_run(self, root: Path, method: str, seed: int) -> Path:
        run_dir = root / f"{method}_s{seed}"
        run_dir.mkdir()
        candidate = method not in {"car_only", "joint_multiview_ce"}
        summary = {
            "probe_seed": seed,
            "probe_objective": "car_only" if method == "car_only" else "multi_view_ce",
            "training_views": ["car"] if method == "car_only" else list(JOINT_TRAINING_VIEWS),
            "validation_prediction_views": list(EVALUATION_VIEWS),
            "consistency_weight": 0.0,
            "validation_predictions_only": True,
            "prediction_split": "audit",
            "all_subject_splits_pairwise_disjoint": True,
            "physionet_test_subjects_used_for_fitting_or_scoring": False,
            "set_queries": 4,
            "train_subjects": list(range(1, 61)),
            "probe_validation_subjects": list(range(61, 71)),
            "audit_subjects": list(range(71, 90)),
            "reserved_test_subjects": list(range(90, 110)),
            "dataset_fingerprint": "fixture-data",
            "encoder_metadata": {
                "model_revision": "model-sha",
                "position_model_revision": "position-sha",
            },
            "probe_auxiliary": GQBA_MODES[method],
            "trainable_parameters": 12345 if candidate else 0,
            "auxiliary_parameters": 2345 if candidate else 0,
            "auxiliary_reference_max_abs_diff": (1e-6 if method in {"gqba_odd", "gqba_odd_even"} else None),
        }
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

        subjects = np.repeat(np.arange(71, 90), 4)
        labels = np.tile(np.arange(4), 19)
        severity = {
            "car_only": 13,
            "joint_multiview_ce": 10,
            "spectral_capacity_control": 9,
            "gqba_odd": 8,
            "gqba_odd_even": 6,
        }[method]
        rows = []
        for view in EVALUATION_VIEWS:
            prediction = labels.copy()
            errors = severity + (2 if view.startswith("native16@") else 0)
            prediction[:errors] = (prediction[:errors] + 1) % 4
            for trial_index, (subject, label, predicted) in enumerate(
                zip(subjects, labels, prediction, strict=True)
            ):
                rows.append(
                    {
                        "defense": "none",
                        "test_view": view,
                        "trial_index": trial_index,
                        "subject_id": subject,
                        "y_true": label,
                        "y_pred": predicted,
                    }
                )
        pd.DataFrame(rows).to_csv(run_dir / "validation_predictions.csv", index=False)
        return run_dir

    def test_complete_screen_writes_frozen_decision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            specs = [f"{method}={self._write_run(root, method, 7)}" for method in GQBA_MODES]
            output = root / "aggregate"
            result = analyze_gqba_benchmark(
                specs,
                output,
                expected_seeds=[7],
                bootstrap_resamples=50,
                bootstrap_seed=5,
            )
            self.assertEqual(set(result["method"]), set(GQBA_MODES))
            summary = json.loads((output / "gqba_summary.json").read_text())
            self.assertEqual(summary["capacity_control"], "spectral_capacity_control")
            self.assertTrue(summary["external_dataset_required_for_confirmation"])
            comparisons = pd.read_csv(output / "gqba_pairwise_bootstrap.csv")
            self.assertIn("both_fists_recall", set(comparisons["metric"]))

    def test_parameter_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            specs = []
            for method in GQBA_MODES:
                path = self._write_run(root, method, 7)
                if method == "gqba_odd":
                    summary_path = path / "summary.json"
                    summary = json.loads(summary_path.read_text())
                    summary["auxiliary_parameters"] += 1
                    summary_path.write_text(json.dumps(summary))
                specs.append(f"{method}={path}")
            with self.assertRaisesRegex(ValueError, "identical parameter counts"):
                analyze_gqba_benchmark(
                    specs,
                    root / "aggregate",
                    expected_seeds=[7],
                    bootstrap_resamples=5,
                )

    def test_cli_exposes_gqba_audit_and_run_mode(self):
        args = build_parser().parse_args(["gqba-audit", "--runs", "car_only=/tmp/example"])
        self.assertEqual(args.command, "gqba-audit")
        run = build_parser().parse_args(
            [
                "run",
                "--config",
                "configs/reve_benchmark_lock.yaml",
                "--probe-auxiliary",
                "gqba_odd_even",
            ]
        )
        self.assertEqual(run.probe_auxiliary, "gqba_odd_even")


if __name__ == "__main__":
    unittest.main()
