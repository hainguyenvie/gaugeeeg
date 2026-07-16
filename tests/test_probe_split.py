import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from gaugeeeg.config import validate_config
from gaugeeeg.datasets import EEGDataset, all_subjects
from gaugeeeg.experiment import run_experiment
from gaugeeeg.montage import SPARSE_MONTAGES


class _FakeEncoder:
    name = "fake"
    cache_signature = "fake:v1"

    def transform(self, x_uv, channel_names, sfreq):
        del channel_names, sfreq
        values = np.asarray(x_uv).mean(axis=2)
        return np.concatenate([values, np.square(values)], axis=1)


class ProbeSplitTests(unittest.TestCase):
    def test_config_requires_audit_split_disjointness(self):
        config = self._config(Path("outputs/test"), Path("outputs/cache"))
        validate_config(config)
        self.assertEqual(all_subjects(config["data"]), list(range(1, 11)))
        config["data"]["audit_subjects"] = [6, 7]
        with self.assertRaisesRegex(ValueError, "pairwise disjoint"):
            validate_config(config)

    def test_validation_only_predictions_use_audit_not_probe_val_or_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root / "run", root / "cache")
            validate_config(config)
            dataset = self._dataset()
            with (
                patch(
                    "gaugeeeg.experiment.load_physionet_mi",
                    return_value=dataset,
                ),
                patch(
                    "gaugeeeg.experiment.build_encoder",
                    return_value=_FakeEncoder(),
                ),
            ):
                run_experiment(config)

            predictions = pd.read_csv(root / "run" / "validation_predictions.csv")
            self.assertEqual(set(predictions["subject_id"]), {7, 8})
            self.assertEqual(set(predictions["split"]), {"audit"})
            summary = json.loads((root / "run" / "summary.json").read_text())
            self.assertTrue(summary["validation_predictions_only"])
            self.assertTrue(summary["all_subject_splits_pairwise_disjoint"])
            self.assertTrue(summary["probe_validation_audit_subjects_disjoint"])
            self.assertFalse(summary["physionet_test_subjects_used_for_fitting_or_scoring"])

    @staticmethod
    def _config(output: Path, cache: Path) -> dict:
        return {
            "seed": 21,
            "data": {
                "train_subjects": [1, 2, 3, 4],
                "val_subjects": [5, 6],
                "audit_subjects": [7, 8],
                "test_subjects": [9, 10],
                "runs": [4, 6],
            },
            "experiment": {
                "encoder": "bandpower",
                "probe": "sklearn_logreg",
                "train_view": "car",
                "training_views": ["car"],
                "test_views": ["car"],
                "validation_prediction_views": [
                    "car",
                    "native16@C3",
                ],
                "defenses": ["none"],
                "probe_seed": 21,
                "reference_seed": 7,
                "save_validation_predictions": True,
                "validation_predictions_only": True,
                "feature_cache_dir": str(cache),
                "output_dir": str(output),
            },
        }

    @staticmethod
    def _dataset() -> EEGDataset:
        subjects = np.repeat(np.arange(1, 11), 8)
        labels = np.tile(np.arange(4), 20)
        x_uv = np.zeros((labels.size, 16, 8), dtype=np.float32)
        x_uv[:, 0, :] = labels[:, None] + 1.0
        x_uv[:, 1, :] = (labels == 0)[:, None] * 0.5
        return EEGDataset(
            x_uv=x_uv,
            y=labels.astype(np.int64),
            subjects=subjects.astype(np.int64),
            channel_names=SPARSE_MONTAGES["sparse16"],
            sfreq=200.0,
            label_names=(
                "left_fist",
                "right_fist",
                "both_fists",
                "both_feet",
            ),
        )


if __name__ == "__main__":
    unittest.main()
