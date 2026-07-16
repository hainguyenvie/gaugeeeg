import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gaugeeeg.class_safeguard import (
    analyze_class_safeguard,
    apply_class_trust_caps,
    fit_class_trust_caps,
)
from gaugeeeg.cli import build_parser
from tests import test_prior_identifiability as prior_fixture


class ClassSafeguardTests(unittest.TestCase):
    def test_class_caps_recover_halfway_source_oracle(self):
        topology = np.asarray(
            [[0.1, -0.2, 0.3], [-0.2, 0.1, -0.1]], dtype=np.float64
        )
        candidate = np.asarray(
            [[0.5, 0.2, -0.1], [0.4, -0.3, 0.2]], dtype=np.float64
        )
        oracle = topology + 0.5 * (candidate - topology)
        caps = fit_class_trust_caps(
            topology,
            candidate,
            oracle,
            n_classes=4,
            ridge=0.0,
        )
        np.testing.assert_allclose(caps, 0.5)

    def test_caps_bound_existing_e11_weight(self):
        topology = np.asarray([0.1, -0.2, 0.3])
        raw_candidate = np.asarray([0.5, 0.2, -0.1])
        e11_weight = 0.6
        e11_candidate = topology + e11_weight * (raw_candidate - topology)
        safe, applied = apply_class_trust_caps(
            topology,
            e11_candidate,
            base_weight=e11_weight,
            class_caps=np.ones(4),
            n_classes=4,
        )
        np.testing.assert_allclose(safe, e11_candidate)
        np.testing.assert_allclose(applied, e11_weight)

        safe, applied = apply_class_trust_caps(
            topology,
            e11_candidate,
            base_weight=e11_weight,
            class_caps=np.zeros(4),
            n_classes=4,
        )
        np.testing.assert_allclose(safe, topology)
        np.testing.assert_allclose(applied, 0.0)

    def test_strict_split_audit_writes_complete_output_suite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "validation_predictions.csv"
            prior_fixture.PriorIdentifiabilityTests._make_predictions().to_csv(
                predictions, index=False
            )
            output = root / "audit"
            aggregate = analyze_class_safeguard(
                predictions,
                output,
                source_subjects=[71, 72],
                adaptation_subjects=[73, 74],
                evaluation_subjects=[81, 82],
                batch_sizes=[4, 8, 16, 32],
                primary_batch_size=4,
                stress_batch_size=8,
                batch_resamples=2,
                source_gate_resamples=1,
                bootstrap_resamples=50,
            )
            summary = json.loads(
                (output / "class_safeguard_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(summary["source_adaptation_subjects_disjoint"])
            self.assertTrue(summary["source_evaluation_subjects_disjoint"])
            self.assertTrue(summary["adaptation_evaluation_subjects_disjoint"])
            self.assertFalse(summary["target_reference_labels_used_for_caps"])
            self.assertIn(
                "safeguard_supported_for_repeated_seed_confirmation", summary
            )
            self.assertIn("paper_level_class_uniform_claim_supported", summary)
            self.assertIn("class_operator_trust_safeguard", set(aggregate["method"]))
            for filename in (
                "class_safeguard_source_examples.csv",
                "class_safeguard_caps.csv",
                "class_safeguard_estimates.csv",
                "class_safeguard_metrics.csv",
                "class_safeguard_aggregate.csv",
                "class_safeguard_bootstrap.csv",
                "class_safeguard_summary.json",
            ):
                self.assertTrue((output / filename).is_file(), filename)
            self.assertTrue(
                (output / "strict_prior_baseline" / "prior_identifiability_summary.json").is_file()
            )

    def test_cli_parser_exposes_strict_split_defaults(self):
        args = build_parser().parse_args(
            [
                "class-safeguard",
                "--validation-predictions",
                "validation.csv",
            ]
        )
        self.assertEqual(args.source_subjects, [71, 72, 73, 74, 75])
        self.assertEqual(args.adaptation_subjects, [76, 77, 78, 79, 80])
        self.assertEqual(args.evaluation_subjects, list(range(81, 90)))
        self.assertEqual(args.source_gate_resamples, 5)


if __name__ == "__main__":
    unittest.main()
