import unittest

from gaugeeeg.datasets import dataset_fingerprint
from gaugeeeg.experiment import _feature_key


class ReproducibilityTests(unittest.TestCase):
    def setUp(self):
        self.data = {
            "train_subjects": [1],
            "val_subjects": [2],
            "test_subjects": [3],
            "runs": [4, 6],
            "resample_hz": 200,
            "epoch_seconds": 4.0,
            "highpass_hz": 0.3,
            "notch_hz": 60.0,
        }

    def test_dataset_fingerprint_changes_with_preprocessing(self):
        original = dataset_fingerprint(self.data)
        changed = dict(self.data, highpass_hz=1.0)
        self.assertNotEqual(original, dataset_fingerprint(changed))

    def test_feature_key_includes_dataset_and_encoder_revisions(self):
        common = {
            "split_name": "train",
            "subjects": [1],
            "view": "car",
            "defense": "none",
            "seed": 7,
        }
        first = _feature_key(
            encoder_signature="reve:model@aaa",
            dataset_signature="data-a",
            **common,
        )
        changed_data = _feature_key(
            encoder_signature="reve:model@aaa",
            dataset_signature="data-b",
            **common,
        )
        changed_model = _feature_key(
            encoder_signature="reve:model@bbb",
            dataset_signature="data-a",
            **common,
        )
        self.assertNotEqual(first, changed_data)
        self.assertNotEqual(first, changed_model)


if __name__ == "__main__":
    unittest.main()
