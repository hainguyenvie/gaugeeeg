import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gaugeeeg.baseline_benchmark import EVALUATION_VIEWS
from gaugeeeg.channel_adaptation_benchmark import (
    CHANNEL_ADAPTATION_SPECS,
    analyze_channel_adaptation_benchmark,
)
from gaugeeeg.cli import build_parser


class ChannelAdaptationBenchmarkTests(unittest.TestCase):
    def _write_run(self, root: Path, method: str, seed: int) -> Path:
        run_dir = root / f"{method}_s{seed}"
        run_dir.mkdir()
        spec, defense = CHANNEL_ADAPTATION_SPECS[method]
        summary = {
            "probe_seed": seed,
            "probe_objective": spec.objective,
            "training_views": list(spec.training_views),
            "validation_prediction_views": list(EVALUATION_VIEWS),
            "consistency_weight": spec.consistency_weight,
            "validation_predictions_only": True,
            "prediction_split": "audit",
            "all_subject_splits_pairwise_disjoint": True,
            "physionet_test_subjects_used_for_fitting_or_scoring": False,
            "set_queries": 4,
            "train_subjects": list(range(1, 61)),
            "probe_validation_subjects": list(range(61, 71)),
            "audit_subjects": list(range(71, 90)),
            "reserved_test_subjects": list(range(90, 110)),
            "dataset_fingerprint": "fixture-data",
            "encoder_metadata": {
                "model_revision": "model-sha",
                "position_model_revision": "position-sha",
            },
        }
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

        rows = []
        subjects = np.repeat(np.arange(71, 90), 4)
        labels = np.tile(np.arange(4), 19)
        method_index = list(CHANNEL_ADAPTATION_SPECS).index(method)
        for view_index, view in enumerate(EVALUATION_VIEWS):
            prediction = labels.copy()
            error_count = max(0, 12 - method_index - view_index // 4)
            prediction[:error_count] = (prediction[:error_count] + 1) % 4
            for trial_index, (subject, label, predicted) in enumerate(
                zip(subjects, labels, prediction, strict=True)
            ):
                rows.append(
                    {
                        "defense": defense,
                        "test_view": view,
                        "trial_index": trial_index,
                        "subject_id": subject,
                        "y_true": label,
                        "y_pred": predicted,
                    }
                )
        pd.DataFrame(rows).to_csv(run_dir / "validation_predictions.csv", index=False)
        return run_dir

    def test_complete_matrix_writes_primary_bootstrap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            specs = [f"{method}={self._write_run(root, method, 7)}" for method in CHANNEL_ADAPTATION_SPECS]
            output = root / "aggregate"
            result = analyze_channel_adaptation_benchmark(
                specs,
                output,
                expected_seeds=[7],
                bootstrap_resamples=50,
                bootstrap_seed=3,
            )
            self.assertEqual(set(result["method"]), set(CHANNEL_ADAPTATION_SPECS))
            pairwise = pd.read_csv(output / "channel_adaptation_pairwise_bootstrap.csv")
            self.assertIn("native16_reference_mean", set(pairwise["test_view"]))
            summary = json.loads((output / "channel_adaptation_summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["adapter_is_gaugeeeg_novel_method"])
            self.assertTrue(summary["external_dataset_required_for_confirmation"])

    def test_cli_exposes_channel_adaptation_audit(self):
        args = build_parser().parse_args(["channel-adaptation-audit", "--runs", "car_only=/tmp/example"])
        self.assertEqual(args.command, "channel-adaptation-audit")


if __name__ == "__main__":
    unittest.main()
