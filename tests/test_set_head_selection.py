import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from gaugeeeg.set_head_selection import select_set_head


class SetHeadSelectionTests(unittest.TestCase):
    def test_selects_on_validation_and_breaks_tie_toward_smaller_head(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = []
            values = {4: (0.50, 0.46), 8: (0.53, 0.47), 16: (0.53, 0.60)}
            for queries, (validation, test) in values.items():
                run = root / f"q{queries}"
                run.mkdir()
                pd.DataFrame(
                    [
                        {
                            "test_view": "car",
                            "probe": "reve_set",
                            "set_queries": queries,
                            "selected_epoch": 3,
                            "validation_balanced_accuracy": validation,
                            "balanced_accuracy": test,
                        }
                    ]
                ).to_csv(run / "metrics.csv", index=False)
                runs.append(run)

            output = root / "selection"
            table = select_set_head(runs, output)
            summary = json.loads((output / "set_head_selection.json").read_text())
            self.assertEqual(summary["selected_queries"], 8)
            self.assertTrue(summary["clean_gate_passed"])
            self.assertEqual(table.loc[table["selected_by_validation"], "set_queries"].item(), 8)

    def test_requires_complete_predeclared_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "q4"
            run.mkdir()
            pd.DataFrame(
                [
                    {
                        "test_view": "car",
                        "probe": "reve_set",
                        "set_queries": 4,
                        "selected_epoch": 1,
                        "validation_balanced_accuracy": 0.5,
                        "balanced_accuracy": 0.5,
                    }
                ]
            ).to_csv(run / "metrics.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Expected exactly one run"):
                select_set_head([run], Path(temporary) / "out")


if __name__ == "__main__":
    unittest.main()
