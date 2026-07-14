import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gaugeeeg.method_compare import aggregate_consistency_methods, compare_consistency_methods


class MethodComparisonTests(unittest.TestCase):
    def test_applies_predeclared_recovery_rule(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = self._make_run(root, "baseline", class_zero_errors=6, cz_bacc=0.50)
            augmentation = self._make_run(root, "augmentation", class_zero_errors=3, cz_bacc=0.55)
            consistency = self._make_run(root, "consistency", class_zero_errors=1, cz_bacc=0.58)
            output = root / "comparison"

            result = compare_consistency_methods(
                baseline,
                augmentation,
                consistency,
                output,
                target_view="cz",
                target_class=0,
                n_resamples=200,
            )
            method = result[result["method"] == "rule_consistency"].iloc[0]
            self.assertGreater(method["target_class_recall_gap_relative_reduction"], 0.30)
            summary = json.loads((output / "method_comparison_summary.json").read_text())
            self.assertTrue(summary["consistency_passes"])
            self.assertTrue(summary["consistency_beats_augmentation_on_primary"])
            paired = pd.read_csv(output / "paired_method_bootstrap.csv")
            primary = paired[paired["metric"] == "target_class_recall_gap_recovery"].iloc[0]
            self.assertGreater(primary["point_estimate"], 0.0)

    def test_aggregates_probe_seeds_with_hierarchical_bootstrap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baselines = []
            augmentations = []
            consistencies = []
            for seed in (7, 21, 42):
                baselines.append(
                    self._make_run(
                        root, f"baseline_{seed}", class_zero_errors=6, cz_bacc=0.50, seed=seed
                    )
                )
                augmentations.append(
                    self._make_run(
                        root, f"augmentation_{seed}", class_zero_errors=3, cz_bacc=0.55, seed=seed
                    )
                )
                consistencies.append(
                    self._make_run(
                        root, f"consistency_{seed}", class_zero_errors=1, cz_bacc=0.58, seed=seed
                    )
                )

            output = root / "aggregate"
            result = aggregate_consistency_methods(
                baselines,
                augmentations,
                consistencies,
                output,
                target_view="cz",
                target_class=0,
                n_resamples=200,
            )
            self.assertEqual(set(result["method"]), {"car_only", "multi_view_ce", "rule_consistency"})
            by_seed = pd.read_csv(output / "method_comparison_by_seed.csv")
            self.assertEqual(set(by_seed["probe_seed"]), {7, 21, 42})
            summary = json.loads((output / "aggregate_method_summary.json").read_text())
            self.assertIn(
                summary["rule_loss_evidence_status"],
                {"supported", "promising_but_inconclusive"},
            )
            self.assertGreater(
                summary["primary_hierarchical_bootstrap"]["point_estimate"], 0.0
            )

    @staticmethod
    def _make_run(
        root: Path,
        name: str,
        *,
        class_zero_errors: int,
        cz_bacc: float,
        seed: int = 7,
    ) -> Path:
        run = root / name
        run.mkdir()
        pd.DataFrame(
            [
                {"probe_seed": seed, "test_view": "car", "balanced_accuracy": 0.60},
                {"probe_seed": seed, "test_view": "cz", "balanced_accuracy": cz_bacc},
            ]
        ).to_csv(run / "metrics.csv", index=False)
        y_true = np.tile(np.arange(4), 10)
        car_prediction = y_true.copy()
        cz_prediction = y_true.copy()
        zero_indices = np.flatnonzero(y_true == 0)[:class_zero_errors]
        cz_prediction[zero_indices] = 2
        frames = []
        for view, prediction in (("car", car_prediction), ("cz", cz_prediction)):
            frames.append(
                pd.DataFrame(
                    {
                        "test_view": view,
                        "trial_index": np.arange(y_true.size),
                        "subject_id": np.repeat(np.arange(4), 10),
                        "probe_seed": seed,
                        "y_true": y_true,
                        "y_pred": prediction,
                    }
                )
            )
        pd.concat(frames).to_csv(run / "predictions.csv", index=False)
        return run


if __name__ == "__main__":
    unittest.main()
