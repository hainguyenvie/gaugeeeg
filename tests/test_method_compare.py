import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gaugeeeg.method_compare import compare_consistency_methods


class MethodComparisonTests(unittest.TestCase):
    def test_applies_predeclared_recovery_rule(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = self._make_run(root, "baseline", class_zero_errors=6, cz_bacc=0.50)
            augmentation = self._make_run(root, "augmentation", class_zero_errors=3, cz_bacc=0.55)
            consistency = self._make_run(root, "consistency", class_zero_errors=1, cz_bacc=0.58)
            output = root / "comparison"

            result = compare_consistency_methods(
                baseline, augmentation, consistency, output, target_view="cz", target_class=0
            )
            method = result[result["method"] == "rule_consistency"].iloc[0]
            self.assertGreater(method["target_class_recall_gap_relative_reduction"], 0.30)
            summary = json.loads((output / "method_comparison_summary.json").read_text())
            self.assertTrue(summary["consistency_passes"])
            self.assertTrue(summary["consistency_beats_augmentation_on_primary"])

    @staticmethod
    def _make_run(root: Path, name: str, *, class_zero_errors: int, cz_bacc: float) -> Path:
        run = root / name
        run.mkdir()
        pd.DataFrame(
            [
                {"test_view": "car", "balanced_accuracy": 0.60},
                {"test_view": "cz", "balanced_accuracy": cz_bacc},
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
                        "y_true": y_true,
                        "y_pred": prediction,
                    }
                )
            )
        pd.concat(frames).to_csv(run / "predictions.csv", index=False)
        return run


if __name__ == "__main__":
    unittest.main()
