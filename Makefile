.PHONY: install test synthetic pilot consistency-multiseed

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
