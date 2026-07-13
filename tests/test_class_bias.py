import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gaugeeeg.class_bias import analyze_class_bias


class ClassBiasTests(unittest.TestCase):
    def test_detects_class_specific_recall_shift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            run.mkdir()
            y_true = np.tile(np.arange(4), 4)
            subjects = np.repeat([90, 91], 8)
            frames = []
            for view in ("car", "cz"):
                prediction = y_true.copy()
                if view == "cz":
                    prediction[y_true == 0] = 2
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
                for class_index, name in enumerate(("left", "right", "both", "feet")):
                    frame[f"prob_{name}"] = (prediction == class_index).astype(float)
                frames.append(frame)
            pd.concat(frames).to_csv(run / "predictions.csv", index=False)

            result = analyze_class_bias([run], root / "output", n_resamples=100, seed=3)
            left = result[(result["test_view"] == "cz") & (result["class_name"] == "left")].iloc[0]
            self.assertEqual(left["recall_gap_mean"], 1.0)
            bootstrap = pd.read_csv(root / "output" / "class_recall_bootstrap.csv")
            left_bootstrap = bootstrap[bootstrap["class_name"] == "left"].iloc[0]
            self.assertEqual(left_bootstrap["ci_lower"], 1.0)


if __name__ == "__main__":
    unittest.main()
