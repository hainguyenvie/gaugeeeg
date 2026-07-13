import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from gaugeeeg.audit import aggregate_audit_runs


class AuditAggregationTests(unittest.TestCase):
    def test_uses_fixed_view_and_sample_standard_deviation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dirs = []
            for seed, car, cz, fcz in [(7, 0.60, 0.55, 0.58), (21, 0.62, 0.56, 0.57)]:
                run_dir = root / f"seed_{seed}"
                run_dir.mkdir()
                run_dirs.append(run_dir)
                pd.DataFrame(
                    [
                        self._row(seed, "car", car, 0.0, 1.0, 1.0),
                        self._row(seed, "cz", cz, car - cz, 0.98, 0.76),
                        self._row(seed, "fcz", fcz, car - fcz, 0.98, 0.77),
                    ]
                ).to_csv(run_dir / "metrics.csv", index=False)

            output_dir = root / "aggregate"
            result = aggregate_audit_runs(run_dirs, output_dir)
            cz = result[result["test_view"] == "cz"].iloc[0]
            self.assertAlmostEqual(cz["gap_from_car_mean"], 0.055)
            self.assertGreater(cz["gap_from_car_std"], 0.0)
            summary = json.loads((output_dir / "aggregate_summary.json").read_text())
            self.assertEqual(summary["std_ddof"], 1)
            self.assertEqual(summary["fixed_worst_reference_view"], "cz")

    @staticmethod
    def _row(seed, view, bacc, gap, cosine, cka):
        return {
            "probe_seed": seed,
            "reference_seed": 7,
            "defense": "none",
            "test_view": view,
            "balanced_accuracy": bacc,
            "balanced_accuracy_gap_from_car": gap,
            "paired_cosine_to_car": cosine,
            "linear_cka_to_car": cka,
        }


if __name__ == "__main__":
    unittest.main()
