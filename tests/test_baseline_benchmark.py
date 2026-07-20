import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gaugeeeg.baseline_benchmark import (
    BASELINE_SPECS,
    EVALUATION_VIEWS,
    analyze_baseline_benchmark,
)
from gaugeeeg.cli import build_parser


class BaselineBenchmarkTests(unittest.TestCase):
    def _write_run(self, root: Path, method: str, seed: int) -> Path:
        run_dir = root / f"{method}_s{seed}"
        run_dir.mkdir()
        spec = BASELINE_SPECS[method]
        summary = {
            "probe_seed": seed,
            "reference_seed": 7,
            "probe_objective": spec.objective,
            "training_views": list(spec.training_views),
            "validation_prediction_views": list(EVALUATION_VIEWS),
            "consistency_weight": spec.consistency_weight,
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
        }
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

        rows = []
        subjects = np.repeat(np.arange(71, 90), 4)
        labels = np.tile(np.arange(4), 19)
        method_index = list(BASELINE_SPECS).index(method)
        for view_index, view in enumerate(EVALUATION_VIEWS):
            prediction = labels.copy()
            # Deterministic, paired errors with different method/view severity.
            error_count = max(0, 16 - method_index - view_index // 4)
            prediction[:error_count] = (prediction[:error_count] + 1) % 4
            for trial_index, (subject, label, predicted) in enumerate(
                zip(subjects, labels, prediction, strict=True)
            ):
                rows.append(
                    {
                        "test_view": view,
                        "trial_index": trial_index,
                        "subject_id": subject,
                        "y_true": label,
                        "y_pred": predicted,
                    }
                )
        pd.DataFrame(rows).to_csv(run_dir / "validation_predictions.csv", index=False)
        return run_dir

    def test_complete_locked_matrix_writes_leaderboard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            specs = []
            for method in BASELINE_SPECS:
                run_dir = self._write_run(root, method, 7)
                specs.append(f"{method}={run_dir}")
            output = root / "aggregate"
            result = analyze_baseline_benchmark(
                specs,
                output,
                expected_seeds=[7],
                bootstrap_resamples=100,
                bootstrap_seed=9,
            )

            self.assertEqual(set(result["method"]), set(BASELINE_SPECS))
            self.assertIn("clean_delta_ci_lower", result.columns)
            self.assertIn("clean_noninferiority_passed", result.columns)
            self.assertTrue((output / "baseline_pairwise_bootstrap.csv").exists())
            summary = json.loads((output / "benchmark_lock_summary.json").read_text())
            self.assertFalse(summary["physionetmi_is_globally_untouched_test"])
            self.assertTrue(summary["external_dataset_required_for_confirmation"])

    def test_rejects_unresolved_encoder_revisions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            specs = []
            for method in BASELINE_SPECS:
                run_dir = self._write_run(root, method, 7)
                summary_path = run_dir / "summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary["encoder_metadata"]["model_revision"] = "unresolved"
                summary["encoder_metadata"]["position_model_revision"] = "unresolved"
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                specs.append(f"{method}={run_dir}")

            with self.assertRaisesRegex(ValueError, "immutable REVE revisions"):
                analyze_baseline_benchmark(
                    specs,
                    root / "aggregate",
                    expected_seeds=[7],
                    bootstrap_resamples=10,
                )

    def test_cli_exposes_locked_baseline_audit(self):
        args = build_parser().parse_args(
            [
                "baseline-benchmark-audit",
                "--runs",
                "car_only=/tmp/example",
            ]
        )
        self.assertEqual(args.command, "baseline-benchmark-audit")


if __name__ == "__main__":
    unittest.main()
