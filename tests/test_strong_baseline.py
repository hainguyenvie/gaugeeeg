import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from gaugeeeg.class_safeguard import (
    BASELINE_METHOD,
    E11_METHOD,
    SAFE_METHOD,
    TOPOLOGY_METHOD,
)
from gaugeeeg.cli import build_parser
from gaugeeeg.strong_baseline import (
    analyze_strong_baselines,
    paired_cluster_bootstrap,
)


class StrongBaselineTests(unittest.TestCase):
    def test_static_topology_is_broadcast_across_candidate_repeats(self):
        rows = []
        for reference, baseline in (("c1", 0.3), ("c2", 0.4)):
            rows.append(
                {
                    "repeat": 0,
                    "held_out_reference": reference,
                    "method": TOPOLOGY_METHOD,
                    "target_bias_rmse": baseline,
                }
            )
            for repeat in (0, 1):
                rows.append(
                    {
                        "repeat": repeat,
                        "held_out_reference": reference,
                        "method": SAFE_METHOD,
                        "target_bias_rmse": baseline - 0.1,
                    }
                )
        result = paired_cluster_bootstrap(
            pd.DataFrame(rows),
            candidate=SAFE_METHOD,
            baseline=TOPOLOGY_METHOD,
            metric="target_bias_rmse",
            n_resamples=100,
            confidence=0.95,
            seed=7,
        )
        self.assertAlmostEqual(result["candidate_minus_baseline"], -0.1)
        self.assertAlmostEqual(result["ci_upper"], -0.1)
        self.assertTrue(result["static_baseline_broadcast_across_repeats"])

    def test_post_hoc_audit_separates_mean_and_class_uniform_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            e12 = root / "e12"
            e12.mkdir()
            summary = {
                "stage": "E12 source-only class/operator trust safeguard",
                "class_names": [
                    "left_fist",
                    "right_fist",
                    "both_fists",
                    "both_feet",
                ],
                "primary_batch_size": 4,
                "stress_batch_size": 8,
                "source_adaptation_subjects_disjoint": True,
                "source_evaluation_subjects_disjoint": True,
                "adaptation_evaluation_subjects_disjoint": True,
                "physionet_test_subjects_used": False,
                "safeguard_supported_for_repeated_seed_confirmation": True,
            }
            (e12 / "class_safeguard_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            self._make_metrics().to_csv(
                e12 / "class_safeguard_metrics.csv", index=False
            )
            output = root / "audit"
            pairwise = analyze_strong_baselines(
                e12,
                output,
                bootstrap_resamples=100,
            )
            result = json.loads(
                (output / "strong_baseline_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(pairwise), 63)
            self.assertFalse(
                result[
                    "mean_rmse_intervals_confirm_vs_all_strong_baselines"
                ]
            )
            self.assertFalse(
                result["mean_method_single_seed_strong_baseline_confirmed"]
            )
            self.assertTrue(result["mean_method_ready_for_new_seed_confirmation"])
            self.assertFalse(
                result["class_uniform_method_ready_for_new_seed_confirmation"]
            )
            self.assertTrue(
                result["class_material_harm_detected_vs_any_strong_baseline"]
            )
            self.assertTrue(result["result_can_falsify_but_not_confirm_a_paper_claim"])
            self.assertTrue((output / "strong_baseline_pairwise.csv").is_file())
            self.assertTrue(
                (output / "strong_baseline_class_audit.csv").is_file()
            )

    def test_cli_exposes_strong_baseline_defaults(self):
        args = build_parser().parse_args(["strong-baseline-audit"])
        self.assertEqual(
            args.e12_output,
            "outputs/reve_set_class_safeguard_audit_s7",
        )
        self.assertEqual(args.bootstrap_resamples, 5000)
        self.assertEqual(args.max_bacc_loss, 0.01)

    @staticmethod
    def _make_metrics() -> pd.DataFrame:
        conditions = [
            ("random", 4, None),
            ("balanced", 8, None),
            *[(f"skew_0.7_class_{index}", 8, index) for index in range(4)],
        ]
        rows = []
        for condition, batch_size, dominant_class in conditions:
            for reference in ("c1", "c2"):
                topology_rmse = 0.25
                topology_gap = 0.10
                rows.append(
                    {
                        "condition": condition,
                        "batch_size": batch_size,
                        "repeat": 0,
                        "target_view": f"native16@{reference}",
                        "held_out_reference": reference,
                        "method": TOPOLOGY_METHOD,
                        "target_bias_rmse": topology_rmse,
                        "balanced_accuracy": 0.80,
                        "max_class_recall_gap_to_car": topology_gap,
                    }
                )
                for repeat in (0, 1):
                    safe_rmse = 0.22 if dominant_class == 1 else 0.10
                    if condition == "random" and reference == "c2":
                        safe_rmse = 0.22
                    e11_rmse = 0.18 if dominant_class == 1 else 0.20
                    safe_gap = 0.13 if dominant_class == 2 else 0.08
                    values = {
                        SAFE_METHOD: (safe_rmse, 0.795, safe_gap),
                        E11_METHOD: (e11_rmse, 0.80, 0.10),
                        BASELINE_METHOD: (0.30, 0.80, 0.12),
                    }
                    for method, (rmse, bacc, gap) in values.items():
                        rows.append(
                            {
                                "condition": condition,
                                "batch_size": batch_size,
                                "repeat": repeat,
                                "target_view": f"native16@{reference}",
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
