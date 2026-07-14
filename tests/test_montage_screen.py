import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gaugeeeg.montage_screen import analyze_montage_screen


class MontageScreenTests(unittest.TestCase):
    def test_writes_fixed_primary_screen_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = {
                name: self._make_run(root / name, seed=7, target_errors=errors)
                for name, errors in {
                    "car_only": 8,
                    "canonical": 7,
                    "augmentation": 5,
                    "consistency": 3,
                }.items()
            }
            output = root / "screen"
            result = analyze_montage_screen(
                runs["car_only"],
                runs["canonical"],
                runs["augmentation"],
                runs["consistency"],
                output,
                n_resamples=100,
            )
            self.assertEqual(
                set(result["method"]),
                {"car_only", "car_canonicalize", "multi_view_ce", "rule_consistency"},
            )
            summary = json.loads((output / "montage_screen_summary.json").read_text())
            self.assertEqual(summary["primary_view"], "sparse16@cz")
            self.assertEqual(summary["selected_consistency_lambda"], 10.0)
            self.assertTrue(summary["lambda_was_selected_before_montage_targets"])
            self.assertTrue((output / "primary_paired_method_bootstrap.csv").exists())

    @staticmethod
    def _make_run(path: Path, *, seed: int, target_errors: int) -> Path:
        path.mkdir()
        y_true = np.tile(np.arange(4), 10)
        car_prediction = y_true.copy()
        target_prediction = y_true.copy()
        target_prediction[:target_errors] = 2
        rows = []
        prediction_frames = []
        for view, prediction, bacc, channels in (
            ("car", car_prediction, 1.0, 64),
            ("sparse16@cz", target_prediction, 1.0 - target_errors / 40.0, 16),
        ):
            rows.append(
                {
                    "probe_seed": seed,
                    "test_view": view,
                    "reference_view": "car" if view == "car" else "cz",
                    "montage": "full" if view == "car" else "sparse16",
                    "n_observed_channels": channels,
                    "balanced_accuracy": bacc,
                }
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "probe_seed": seed,
                        "test_view": view,
                        "trial_index": np.arange(y_true.size),
                        "subject_id": np.repeat(np.arange(4), 10),
                        "y_true": y_true,
                        "y_pred": prediction,
                    }
                )
            )
        pd.DataFrame(rows).to_csv(path / "metrics.csv", index=False)
        pd.concat(prediction_frames, ignore_index=True).to_csv(path / "predictions.csv", index=False)
        return path


if __name__ == "__main__":
    unittest.main()
