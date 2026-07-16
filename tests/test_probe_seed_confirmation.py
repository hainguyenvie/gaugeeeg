import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from gaugeeeg.class_safeguard import E11_METHOD, SAFE_METHOD, TOPOLOGY_METHOD
from gaugeeeg.cli import build_parser
from gaugeeeg.probe_seed_confirmation import analyze_probe_seed_confirmation


class ProbeSeedConfirmationTests(unittest.TestCase):
    def test_untouched_seed_confirmation_separates_mean_and_class_claims(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = self._write_frozen_gate(root)
            logit_runs = []
            e12_runs = []
            for seed in (21, 42):
                logit_run, e12_run = self._write_seed(root, seed)
                logit_runs.append(logit_run)
                e12_runs.append(e12_run)

            output = root / "confirmation"
            result = analyze_probe_seed_confirmation(
                frozen,
                logit_runs,
                e12_runs,
                output,
                bootstrap_resamples=100,
            )
            summary = json.loads((output / "probe_seed_confirmation_summary.json").read_text())
            self.assertEqual(len(result), 42)
            self.assertEqual(summary["new_probe_seeds"], [21, 42])
            self.assertTrue(summary["mean_method_new_seed_confirmation_supported"])
            self.assertFalse(summary["new_seed_class_uniform_diagnostic_supported"])
            self.assertTrue(summary["current_class_uniform_claim_remains_rejected"])
            self.assertFalse(summary["paper_level_mean_claim_supported"])
            self.assertEqual(
                summary["next_method_recommendation"],
                "validate_frozen_mean_method_on_external_open_eeg_dataset",
            )
            self.assertTrue((output / "probe_seed_manifest.csv").is_file())
            self.assertTrue((output / "probe_seed_hierarchical_pairwise.csv").is_file())

    def test_rejects_a_probe_run_that_touched_test_subjects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = self._write_frozen_gate(root)
            pairs = [self._write_seed(root, seed) for seed in (21, 42)]
            bad_summary_path = pairs[0][0] / "summary.json"
            bad = json.loads(bad_summary_path.read_text())
            bad["physionet_test_subjects_used_for_fitting_or_scoring"] = True
            bad_summary_path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "protocol failed"):
                analyze_probe_seed_confirmation(
                    frozen,
                    [pair[0] for pair in pairs],
                    [pair[1] for pair in pairs],
                    root / "confirmation",
                    bootstrap_resamples=10,
                )

    def test_cli_defaults_to_two_untouched_seeds(self):
        args = build_parser().parse_args(["confirm-probe-seeds"])
        self.assertEqual(args.exploratory_probe_seed, 7)
        self.assertEqual(args.bootstrap_resamples, 10000)
        self.assertEqual(len(args.logit_runs), 2)
        self.assertEqual(len(args.e12_runs), 2)

    @staticmethod
    def _write_frozen_gate(root: Path) -> Path:
        path = root / "e13.json"
        path.write_text(
            json.dumps(
                {
                    "stage": "E13 post-hoc strongest-baseline audit",
                    "audit_status": "post_hoc_falsification_only",
                    "mean_method_ready_for_new_seed_confirmation": True,
                    "class_uniform_method_ready_for_new_seed_confirmation": False,
                    "predeclared_strong_baselines": [
                        E11_METHOD,
                        TOPOLOGY_METHOD,
                    ],
                    "max_bacc_loss": 0.01,
                    "max_gap_increase": 0.01,
                }
            )
        )
        return path

    @classmethod
    def _write_seed(cls, root: Path, seed: int) -> tuple[Path, Path]:
        logit_run = root / f"logits_s{seed}"
        e12_run = root / f"e12_s{seed}"
        logit_run.mkdir()
        e12_run.mkdir()
        (logit_run / "summary.json").write_text(
            json.dumps(
                {
                    "probe_seed": seed,
                    "reference_seed": 7,
                    "encoder": "reve",
                    "probe": "reve_set",
                    "probe_objective": "car_only",
                    "set_queries": 4,
                    "set_heads": 8,
                    "strict_determinism": True,
                    "validation_predictions_only": True,
                    "prediction_split": "audit",
                    "all_subject_splits_pairwise_disjoint": True,
                    "probe_validation_audit_subjects_disjoint": True,
                    "physionet_test_subjects_used_for_fitting_or_scoring": False,
                    "audit_subjects": list(range(71, 90)),
                    "train_subjects": list(range(1, 61)),
                    "probe_validation_subjects": list(range(61, 71)),
                    "reserved_test_subjects": list(range(90, 110)),
                    "validation_balanced_accuracy": 0.5,
                    "selected_epoch": 4,
                }
            )
        )
        (e12_run / "class_safeguard_summary.json").write_text(
            json.dumps(
                {
                    "stage": "E12 source-only class/operator trust safeguard",
                    "probe_seed": seed,
                    "physionet_test_subjects_used": False,
                    "source_adaptation_subjects_disjoint": True,
                    "source_evaluation_subjects_disjoint": True,
                    "adaptation_evaluation_subjects_disjoint": True,
                    "source_subjects": list(range(71, 76)),
                    "adaptation_subjects": list(range(76, 81)),
                    "evaluation_subjects": list(range(81, 90)),
                    "class_names": [
                        "left_fist",
                        "right_fist",
                        "both_fists",
                        "both_feet",
                    ],
                    "primary_batch_size": 4,
                    "stress_batch_size": 8,
                    "safeguard_supported_for_repeated_seed_confirmation": True,
                }
            )
        )
        cls._metrics().to_csv(e12_run / "class_safeguard_metrics.csv", index=False)
        return logit_run, e12_run

    @staticmethod
    def _metrics() -> pd.DataFrame:
        conditions = [
            ("random", 4, None),
            ("balanced", 8, None),
            *[(f"skew_0.7_class_{index}", 8, index) for index in range(4)],
        ]
        rows = []
        for condition, batch_size, dominant_class in conditions:
            for reference in ("c1", "c2"):
                rows.append(
                    {
                        "condition": condition,
                        "batch_size": batch_size,
                        "repeat": 0,
                        "held_out_reference": reference,
                        "method": TOPOLOGY_METHOD,
                        "target_bias_rmse": 0.25,
                        "balanced_accuracy": 0.80,
                        "max_class_recall_gap_to_car": 0.10,
                    }
                )
                for repeat in (0, 1):
                    safe_rmse = 0.22 if dominant_class == 1 else 0.10
                    for method, rmse, bacc, gap in (
                        (SAFE_METHOD, safe_rmse, 0.795, 0.09),
                        (E11_METHOD, 0.20, 0.80, 0.10),
                    ):
                        rows.append(
                            {
                                "condition": condition,
                                "batch_size": batch_size,
                                "repeat": repeat,
                                "held_out_reference": reference,
                                "method": method,
                                "target_bias_rmse": rmse,
                                "balanced_accuracy": bacc,
                                "max_class_recall_gap_to_car": gap,
                            }
                        )
        return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
