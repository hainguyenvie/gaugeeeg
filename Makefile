.PHONY: install test synthetic pilot consistency-multiseed consistency-lambda-ablation montage-screen native-montage-screen set-native-screen set-reference-closure set-reference-geometry set-calibration-control set-bias-manifold set-prior-stress set-prior-identifiability set-class-safeguard set-strong-baseline-audit set-probe-seed-confirmation set-operator-consistency benchmark-baselines

install:
	python -m pip install -e ".[data,dev]"

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

synthetic:
	PYTHONPATH=src python -m gaugeeeg.synthetic

pilot:
	PYTHONPATH=src python -m gaugeeeg.cli run --config configs/pilot.yaml --encoder bandpower

consistency-multiseed:
	bash scripts/run_consistency_multiseed.sh

consistency-lambda-ablation:
	bash scripts/run_consistency_lambda_ablation.sh

montage-screen:
	bash scripts/run_montage_screen.sh

native-montage-screen:
	bash scripts/run_native_montage_screen.sh

set-native-screen:
	bash scripts/run_reve_set_native_screen.sh

set-reference-closure:
	bash scripts/run_reve_set_reference_closure.sh

set-reference-geometry:
	bash scripts/run_reve_set_reference_geometry.sh

set-calibration-control:
	bash scripts/run_reve_set_calibration_control.sh

set-bias-manifold:
	bash scripts/run_reve_set_bias_manifold.sh

set-prior-stress:
	bash scripts/run_reve_set_prior_stress.sh

set-prior-identifiability:
	bash scripts/run_reve_set_prior_identifiability.sh

set-class-safeguard:
	bash scripts/run_reve_set_class_safeguard.sh

set-strong-baseline-audit:
	bash scripts/run_reve_set_strong_baseline_audit.sh

set-probe-seed-confirmation:
	bash scripts/run_reve_set_probe_seed_confirmation.sh

set-operator-consistency:
	bash scripts/run_reve_set_operator_consistency.sh

benchmark-baselines:
	bash scripts/run_reve_benchmark_baselines.sh
