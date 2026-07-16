import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from gaugeeeg.cli import build_parser
from gaugeeeg.config import load_config, validate_config
from gaugeeeg.operator_consistency import (
    AUDIT_VIEWS,
    CAR_ONLY,
    MULTI_VIEW,
    OPERATOR,
    TRAINING_VIEWS,
    analyze_operator_consistency,
)


class OperatorConsistencyTests(unittest.TestCase):
    def test_config_and_cli_expose_variable_set_operator_objective(self):
        config = load_config("configs/reve_set_operator_consistency_q4.yaml")
        experiment = config["experiment"]
        self.assertEqual(experiment["probe"], "reve_set")
        self.assertEqual(experiment["probe_objective"], OPERATOR)
        self.assertEqual(experiment["training_views"], list(TRAINING_VIEWS))
        self.assertEqual(experiment["consistency_view_weights"], [0.0, 0.5, 1.0])

        changed = deepcopy(config)
        changed["experiment"]["consistency_view_weights"] = [0.0, 1.0]
        with self.assertRaisesRegex(ValueError, "one consistency_view_weight"):
            validate_config(changed)

        changed = deepcopy(config)
        changed["experiment"]["probe"] = "reve_token"
        with self.assertRaisesRegex(ValueError, "requires probe: reve_set"):
            validate_config(changed)

        args = build_parser().parse_args(
            [
                "operator-consistency-audit",
                "--car-only-run",
                "car",
                "--multi-view-run",
                "multi",
                "--operator-run",
                "operator",
            ]
        )
        self.assertEqual(args.minimum_native16_bacc_gain, 0.02)
        self.assertEqual(args.maximum_clean_bacc_loss, 0.01)

    def test_development_gate_uses_audit_only_and_keeps_test_locked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            e14 = root / "e14.json"
            e14.write_text(
                json.dumps(
                    {
                        "stage": "E14 untouched-probe-seed mean-method confirmation",
                        "mean_method_new_seed_confirmation_supported": False,
                        "physionet_test_subjects_used": False,
                        "next_method_recommendation": (
                            "do_not_tune_on_audit_subjects_revisit_source_only_method"
                        ),
                    }
                ),
                encoding="utf-8",
            )
            run_dirs = {}
            for method in (CAR_ONLY, MULTI_VIEW, OPERATOR):
                run_dir = root / method
                self._write_run(run_dir, method)
                run_dirs[method] = run_dir

            output = root / "audit"
            analyze_operator_consistency(
                e14,
                run_dirs[CAR_ONLY],
                run_dirs[MULTI_VIEW],
                run_dirs[OPERATOR],
                output,
                bootstrap_resamples=100,
            )
            summary = json.loads(
                (output / "operator_consistency_summary.json").read_text()
            )
            self.assertTrue(summary["clean_car_preserved"])
            self.assertTrue(summary["native16_recovery_vs_car_only_supported"])
            self.assertTrue(summary["operator_rule_beats_multi_view_ce"])
            self.assertTrue(
                summary["operator_consistency_development_gate_supported"]
            )
            self.assertFalse(summary["physionet_test_subjects_used"])
            self.assertFalse(summary["paper_level_claim_supported"])
            self.assertEqual(
                summary["next_method_recommendation"],
                "freeze_e15_and_run_reserved_test_multiseed_once",
            )
            for filename in (
                "operator_consistency_by_view.csv",
                "operator_consistency_pairwise.csv",
                "operator_consistency_manifest.csv",
            ):
                self.assertTrue((output / filename).is_file(), filename)

            operator_summary_path = run_dirs[OPERATOR] / "summary.json"
            operator_summary = json.loads(operator_summary_path.read_text())
            operator_summary[
                "physionet_test_subjects_used_for_fitting_or_scoring"
            ] = True
            operator_summary_path.write_text(json.dumps(operator_summary))
            with self.assertRaisesRegex(ValueError, "test_untouched"):
                analyze_operator_consistency(
                    e14,
                    run_dirs[CAR_ONLY],
                    run_dirs[MULTI_VIEW],
                    run_dirs[OPERATOR],
                    root / "rejected",
                    bootstrap_resamples=10,
                )

    @staticmethod
    def _write_run(run_dir: Path, method: str) -> None:
        run_dir.mkdir(parents=True)
        training_views = ["car"] if method == CAR_ONLY else list(TRAINING_VIEWS)
        summary = {
            "encoder": "reve",
            "probe": "reve_set",
            "set_queries": 4,
            "probe_objective": method,
            "training_views": training_views,
            "validation_prediction_views": list(AUDIT_VIEWS),
            "strict_determinism": True,
            "probe_seed": 7,
            "reference_seed": 7,
            "validation_predictions_only": True,
            "prediction_split": "audit",
            "all_subject_splits_pairwise_disjoint": True,
            "physionet_test_subjects_used_for_fitting_or_scoring": False,
            "train_subjects": list(range(1, 61)),
            "probe_validation_subjects": list(range(61, 71)),
            "audit_subjects": list(range(71, 90)),
            "reserved_test_subjects": list(range(90, 110)),
            "selected_epoch": 3,
            "validation_balanced_accuracy": 0.75,
            "consistency_weight": 1.0 if method == OPERATOR else 0.0,
            "consistency_view_weights": [0.0, 0.5, 1.0],
        }
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

        labels = np.tile(np.arange(4), 19)
        subjects = np.repeat(np.arange(71, 90), 4)
        trial_index = np.arange(labels.size)
        frames = []
        for view in AUDIT_VIEWS:
            if view == "native16@car":
                patterns = {
                    CAR_ONLY: np.asarray([1, 1, 3, 3]),
                    MULTI_VIEW: np.asarray([0, 1, 3, 3]),
                    OPERATOR: np.asarray([0, 1, 2, 3]),
                }
            elif view == "native32@car":
                patterns = {
                    CAR_ONLY: np.asarray([1, 1, 2, 3]),
                    MULTI_VIEW: np.asarray([0, 1, 2, 3]),
                    OPERATOR: np.asarray([0, 1, 2, 3]),
                }
            else:
                patterns = {
                    CAR_ONLY: np.asarray([0, 1, 2, 3]),
                    MULTI_VIEW: np.asarray([0, 1, 2, 3]),
                    OPERATOR: np.asarray([0, 1, 2, 3]),
                }
            predictions = np.tile(patterns[method], 19)
            frames.append(
                pd.DataFrame(
                    {
                        "split": "audit",
                        "test_view": view,
                        "trial_index": trial_index,
                        "subject_id": subjects,
                        "y_true": labels,
                        "y_pred": predictions,
                    }
                )
            )
        pd.concat(frames, ignore_index=True).to_csv(
            run_dir / "validation_predictions.csv", index=False
        )


if __name__ == "__main__":
    unittest.main()
