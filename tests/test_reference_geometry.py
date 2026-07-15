import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gaugeeeg.reference_geometry import analyze_reference_geometry


class ReferenceGeometryTests(unittest.TestCase):
    def test_supports_joint_scope_from_predeclared_native_pz_class_shift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "geometry"
            full = root / "full"
            run.mkdir()
            full.mkdir()
            y_true = np.tile(np.arange(4), 10)
            subjects = np.repeat(np.arange(90, 94), 10)
            car = y_true.copy()
            predictions = {"car": car}
            for size in (32, 16):
                base = y_true.copy()
                predictions[f"native{size}@car"] = base
                predictions[f"native{size}@cz"] = base.copy()
                pz = base.copy()
                pz[y_true == 0] = 1
                predictions[f"native{size}@pz"] = pz
                predictions[f"native{size}@fz"] = base.copy()

            self._write_predictions(run, predictions, y_true, subjects)
            self._write_predictions(full, {"car": car}, y_true, subjects)
            metric_rows = []
            for view in predictions:
                metric_rows.append(
                    {
                        "test_view": view,
                        "set_queries": 4,
                        "macro_auroc_ovr": 0.8,
                        "paired_cosine_to_car": 0.9,
                        "linear_cka_to_car": 0.8,
                    }
                )
            pd.DataFrame(metric_rows).to_csv(run / "metrics.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "test_view": "car",
                        "balanced_accuracy_gap_from_car": 0.0,
                    },
                    {
                        "test_view": "pz",
                        "balanced_accuracy_gap_from_car": 0.04,
                    },
                ]
            ).to_csv(full / "metrics.csv", index=False)
            selection = root / "selection.json"
            selection.write_text(json.dumps({"selected_queries": 4}), encoding="utf-8")

            output = root / "audit"
            result = analyze_reference_geometry(
                run,
                full,
                selection,
                output,
                n_resamples=50,
                bootstrap_seed=11,
            )
            summary = json.loads((output / "reference_geometry_summary.json").read_text())
            self.assertEqual(len(result), 6)
            self.assertTrue(summary["car_predictions_reproduced_exactly"])
            self.assertTrue(summary["native16_alternative_aggregate_shift_supported"])
            self.assertTrue(summary["native16_alternative_class_shift_supported"])
            self.assertTrue(summary["joint_gauge_montage_target_supported"])
            self.assertEqual(summary["method_scope_decision"], "joint_gauge_montage_learning")

    @staticmethod
    def _write_predictions(
        run_dir: Path,
        predictions: dict[str, np.ndarray],
        y_true: np.ndarray,
        subjects: np.ndarray,
    ) -> None:
        frames = []
        names = ("left_fist", "right_fist", "both_fists", "both_feet")
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
            for class_index, name in enumerate(names):
                frame[f"prob_{name}"] = (prediction == class_index).astype(float)
            frames.append(frame)
        pd.concat(frames, ignore_index=True).to_csv(run_dir / "predictions.csv", index=False)


if __name__ == "__main__":
    unittest.main()
