import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gaugeeeg.reference_closure import analyze_reference_closure


class ReferenceClosureTests(unittest.TestCase):
    def test_closes_full_reference_and_detects_functional_class_collapse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            full_run = root / "full"
            native_run = root / "native"
            full_run.mkdir()
            native_run.mkdir()
            y_true = np.tile(np.arange(4), 10)
            subjects = np.repeat(np.arange(90, 94), 10)

            full_predictions = {
                "car": y_true.copy(),
                "cz": np.where(y_true == 0, 2, y_true),
                "pz": y_true.copy(),
                "fcz": np.where(y_true == 1, 3, y_true),
            }
            native16_car = y_true.copy()
            native16_car[y_true == 2] = 0
            native16_car[y_true == 3] = 1
            native16_cz = native16_car.copy()
            native_predictions = {
                "car": y_true.copy(),
                "native16@car": native16_car,
                "native16@cz": native16_cz,
            }
            self._write_predictions(full_run, full_predictions, y_true, subjects)
            self._write_predictions(native_run, native_predictions, y_true, subjects)
            pd.DataFrame(
                [{"test_view": view, "set_queries": 4} for view in full_predictions]
            ).to_csv(full_run / "metrics.csv", index=False)
            pd.DataFrame(
                [{"test_view": view, "set_queries": 4} for view in native_predictions]
            ).to_csv(native_run / "metrics.csv", index=False)
            selection = root / "selection.json"
            selection.write_text(json.dumps({"selected_queries": 4}), encoding="utf-8")

            output = root / "closure"
            result = analyze_reference_closure(
                full_run,
                native_run,
                selection,
                output,
                n_resamples=50,
                bootstrap_seed=3,
            )
            summary = json.loads((output / "reference_closure_summary.json").read_text())
            self.assertEqual(len(result), 5)
            self.assertTrue(summary["car_predictions_reproduced_exactly"])
            self.assertTrue(summary["functional_two_class_collapse"])
            self.assertTrue(summary["collapsed_class_ranking_signal_preserved"])
            self.assertEqual(summary["collapsed_class_indices"], [2, 3])
            self.assertIn("recoverable_ranking_signal", summary["audit_status"])
            self.assertTrue((output / "class_conditional_metrics.csv").exists())

    @staticmethod
    def _write_predictions(
        run_dir: Path,
        predictions: dict[str, np.ndarray],
        y_true: np.ndarray,
        subjects: np.ndarray,
    ) -> None:
        frames = []
        for view, prediction in predictions.items():
            frame = pd.DataFrame(
                {
                    "probe_seed": 7,
                    "test_view": view,
                    "trial_index": np.arange(y_true.size),
                    "subject_id": subjects,
                    "y_true": y_true,
                    "y_pred": prediction,
                }
            )
            for class_index, class_name in enumerate(
                ("left_fist", "right_fist", "both_fists", "both_feet")
            ):
                score = np.full(y_true.size, 0.05)
                score[y_true == class_index] = 0.25
                score[prediction == class_index] = 0.70
                frame[f"prob_{class_name}"] = score
            frames.append(frame)
        pd.concat(frames, ignore_index=True).to_csv(run_dir / "predictions.csv", index=False)


if __name__ == "__main__":
    unittest.main()
