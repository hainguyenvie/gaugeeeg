import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gaugeeeg.lambda_ablation import analyze_lambda_ablation


class LambdaAblationTests(unittest.TestCase):
    def test_selects_lambda_from_validation_before_test_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seeds = (7, 21, 42)
            weights = (0.0, 0.3, 1.0, 3.0, 10.0)
            validation = {0.0: 0.50, 0.3: 0.52, 1.0: 0.54, 3.0: 0.57, 10.0: 0.55}
            class_errors = {0.0: 4, 0.3: 3, 1.0: 2, 3.0: 1, 10.0: 1}
            baselines = []
            runs = []
            for seed in seeds:
                baselines.append(
                    self._make_run(
                        root,
                        f"baseline_{seed}",
                        seed=seed,
                        objective="car_only",
                        weight=0.0,
                        validation=0.48,
                        class_zero_errors=6,
                        cz_bacc=0.50,
                    )
                )
                for weight in weights:
                    tag = str(weight).replace(".", "p")
                    runs.append(
                        self._make_run(
                            root,
                            f"lambda_{tag}_{seed}",
                            seed=seed,
                            objective="multi_view_ce" if weight == 0.0 else "rule_consistency",
                            weight=weight,
                            validation=validation[weight] + seed * 1e-5,
                            class_zero_errors=class_errors[weight],
                            cz_bacc=0.55 + 0.005 * min(weight, 3.0),
                        )
                    )

            output = root / "lambda_ablation"
            result = analyze_lambda_ablation(
                baselines,
                runs,
                output,
                expected_lambdas=list(weights),
                target_view="cz",
                target_class=0,
                n_resamples=200,
            )
            selected = result.sort_values("validation_balanced_accuracy_mean").iloc[-1]
            self.assertEqual(selected["lambda"], 3.0)
            summary = json.loads((output / "lambda_ablation_summary.json").read_text())
            self.assertEqual(summary["selected_lambda"], 3.0)
            self.assertFalse(summary["selection_uses_target_view"])
            self.assertIsNotNone(summary["selected_vs_augmentation"])
            self.assertTrue((output / "lambda_test_ablation_summary.csv").exists())

    @staticmethod
    def _make_run(
        root: Path,
        name: str,
        *,
        seed: int,
        objective: str,
        weight: float,
        validation: float,
        class_zero_errors: int,
        cz_bacc: float,
    ) -> Path:
        run = root / name
        run.mkdir()
        pd.DataFrame(
            [
                {
                    "probe_seed": seed,
                    "probe_objective": objective,
                    "consistency_weight": weight,
                    "selected_epoch": 4,
                    "validation_balanced_accuracy": validation,
                    "validation_consistency_loss": 0.02 / (1.0 + weight),
                    "validation_prediction_disagreement": 0.10 / (1.0 + weight),
                    "test_view": "car",
                    "balanced_accuracy": 0.60,
                },
                {
                    "probe_seed": seed,
                    "probe_objective": objective,
                    "consistency_weight": weight,
                    "selected_epoch": 4,
                    "validation_balanced_accuracy": validation,
                    "validation_consistency_loss": 0.02 / (1.0 + weight),
                    "validation_prediction_disagreement": 0.10 / (1.0 + weight),
                    "test_view": "cz",
                    "balanced_accuracy": cz_bacc,
                },
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
                        "probe_seed": seed,
                        "test_view": view,
                        "trial_index": np.arange(y_true.size),
                        "subject_id": np.repeat(np.arange(4), 10),
                        "y_true": y_true,
                        "y_pred": prediction,
                    }
                )
            )
        pd.concat(frames, ignore_index=True).to_csv(run / "predictions.csv", index=False)
        return run


if __name__ == "__main__":
    unittest.main()
