.PHONY: install test synthetic pilot consistency-multiseed consistency-lambda-ablation montage-screen native-montage-screen set-native-screen set-reference-closure set-reference-geometry set-calibration-control set-bias-manifold

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
