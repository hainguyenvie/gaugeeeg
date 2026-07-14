import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from gaugeeeg.native_montage_screen import analyze_native_montage_screen


class NativeMontageScreenTests(unittest.TestCase):
    def test_accepts_hard_noncollapsed_native_screen(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = self._make_run(root / "baseline", canonical=False)
            canonical = self._make_run(root / "canonical", canonical=True)
            output = root / "screen"
            result = analyze_native_montage_screen(
                baseline,
                canonical,
                output,
                n_resamples=100,
            )
            self.assertEqual(set(result["method"]), {"full_car_probe", "car_canonicalize"})
            summary = json.loads((output / "native_montage_screen_summary.json").read_text())
            self.assertEqual(summary["benchmark_status"], "usable_native_montage_benchmark")
            self.assertTrue(summary["primary_prediction_noncollapse_passed"])
            self.assertEqual(summary["canonical_reference_residual_within_montage"], 0.0)

    @staticmethod
    def _make_run(path: Path, *, canonical: bool) -> Path:
        path.mkdir()
        y_true = np.tile(np.arange(4), 10)
        subjects = np.repeat(np.arange(4), 10)
        car = y_true.copy()

        def with_errors(count: int) -> np.ndarray:
            prediction = y_true.copy()
            indices = np.arange(count)
            prediction[indices] = (prediction[indices] + 1) % 4
            return prediction

        predictions = {
            "car": car,
            "native32@car": with_errors(4),
            "native16@car": with_errors(8),
            "native8@car": with_errors(12),
            "native32@cz": with_errors(5),
            "native16@cz": with_errors(8 if canonical else 10),
            "native8@cz": with_errors(14),
        }
        metric_rows = []
        prediction_frames = []
        for view, prediction in predictions.items():
            size = 64 if view == "car" else int(view.split("@", maxsplit=1)[0].replace("native", ""))
            metric_rows.append(
                {
                    "probe_seed": 7,
                    "test_view": view,
                    "reference_view": "car" if "@car" in view or view == "car" else "cz",
                    "montage": "full" if view == "car" else f"sparse{size}",
                    "n_observed_channels": size,
                    "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
                    "macro_f1": f1_score(y_true, prediction, average="macro"),
                    "macro_auroc_ovr": 0.8,
                    "paired_cosine_to_car": 0.9,
                    "linear_cka_to_car": 0.8,
                }
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "test_view": view,
                        "trial_index": np.arange(y_true.size),
                        "subject_id": subjects,
                        "y_true": y_true,
                        "y_pred": prediction,
                    }
                )
            )
        pd.DataFrame(metric_rows).to_csv(path / "metrics.csv", index=False)
        pd.concat(prediction_frames, ignore_index=True).to_csv(path / "predictions.csv", index=False)
        return path


if __name__ == "__main__":
    unittest.main()
