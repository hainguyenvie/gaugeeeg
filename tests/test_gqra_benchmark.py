import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gaugeeeg.baseline_benchmark import EVALUATION_VIEWS, JOINT_TRAINING_VIEWS
from gaugeeeg.cli import build_parser
from gaugeeeg.config import load_config, validate_config
from gaugeeeg.gqra_benchmark import GQRA_SPECS, NEW_METHODS, analyze_gqra_benchmark


class GQRABenchmarkTests(unittest.TestCase):
    def _write_run(self, root: Path, method: str, seed: int) -> Path:
        run_dir = root / f"{method}_s{seed}"
        run_dir.mkdir()
        spec = GQRA_SPECS[method]
        new_method = method in NEW_METHODS
        summary = {
            "probe_seed": seed,
            "probe_objective": "multi_view_ce",
            "training_views": list(JOINT_TRAINING_VIEWS),
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
            "probe_auxiliary": spec.auxiliary,
            "auxiliary_fusion": spec.fusion,
            "auxiliary_preservation_weight": spec.preservation_weight,
            "auxiliary_residual_consistency_weight": spec.residual_consistency_weight,
            "auxiliary_gate_supervision_weight": spec.gate_supervision_weight,
            "representation_contrastive_weight": spec.contrastive_weight,
            "representation_bilaterality_weight": spec.bilaterality_weight,
            "representation_temperature": spec.temperature,
            "auxiliary_target_classes": [2, 3] if method != "joint_multiview_ce" else [],
            "trainable_parameters": 18000 if new_method else 15000,
            "auxiliary_parameters": 7000 if new_method else (4000 if method == "gsra" else 0),
            "auxiliary_reference_max_abs_diff": (1e-6 if spec.auxiliary == "gqba_odd_even" else None),
            "validation_representation_alignment_loss": 0.05 if new_method else None,
            "validation_representation_class_margin": 0.2 if new_method else None,
            "validation_representation_bilaterality_balanced_accuracy": (0.7 if new_method else None),
        }
        if method == "gqba_film_ce":
            summary["validation_representation_alignment_loss"] = 0.08
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

        subjects = np.repeat(np.arange(71, 90), 4)
        labels = np.tile(np.arange(4), 19)
        severity = {
            "joint_multiview_ce": 12,
            "gsra": 11,
            "film_spectral_control": 9,
            "gqba_film_ce": 8,
            "gqra": 4,
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

    def test_complete_screen_writes_locked_decision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            specs = [f"{method}={self._write_run(root, method, 7)}" for method in GQRA_SPECS]
            output = root / "aggregate"
            result = analyze_gqra_benchmark(
                specs,
                output,
                expected_seeds=[7],
                bootstrap_resamples=50,
                bootstrap_seed=5,
            )
            self.assertEqual(set(result["method"]), set(GQRA_SPECS))
            summary = json.loads((output / "gqra_summary.json").read_text())
            self.assertEqual(summary["proposed_method"], "gqra")
            self.assertEqual(summary["capacity_control"], "film_spectral_control")
            self.assertTrue(summary["external_dataset_required_for_confirmation"])
            comparisons = pd.read_csv(output / "gqra_pairwise_bootstrap.csv")
            self.assertIn("class_2_recall", set(comparisons["metric"]))

    def test_film_parameter_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            specs = []
            for method in GQRA_SPECS:
                path = self._write_run(root, method, 7)
                if method == "gqra":
                    summary_path = path / "summary.json"
                    summary = json.loads(summary_path.read_text())
                    summary["auxiliary_parameters"] += 1
                    summary_path.write_text(json.dumps(summary))
                specs.append(f"{method}={path}")
            with self.assertRaisesRegex(ValueError, "identical parameter counts"):
                analyze_gqra_benchmark(
                    specs,
                    root / "aggregate",
                    expected_seeds=[7],
                    bootstrap_resamples=5,
                )

    def test_cli_and_config_expose_gqra_controls(self):
        args = build_parser().parse_args(["gqra-audit", "--runs", "gqra=/tmp/example"])
        self.assertEqual(args.command, "gqra-audit")
        run = build_parser().parse_args(
            [
                "run",
                "--config",
                "configs/reve_benchmark_lock.yaml",
                "--probe-auxiliary",
                "gqba_odd_even",
                "--auxiliary-fusion",
                "film",
                "--representation-contrastive-weight",
                "0.1",
                "--representation-bilaterality-weight",
                "0.2",
                "--representation-temperature",
                "0.1",
            ]
        )
        self.assertEqual(run.auxiliary_fusion, "film")
        self.assertEqual(run.representation_contrastive_weight, 0.1)

        config = load_config("configs/reve_benchmark_lock.yaml")
        changed = json.loads(json.dumps(config))
        changed["experiment"]["representation_contrastive_weight"] = 0.1
        with self.assertRaisesRegex(ValueError, "requires auxiliary_fusion: film"):
            validate_config(changed)


if __name__ == "__main__":
    unittest.main()
