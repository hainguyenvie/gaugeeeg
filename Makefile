.PHONY: install test synthetic pilot

install:
	python -m pip install -e ".[data,dev]"

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

synthetic:
	PYTHONPATH=src python -m gaugeeeg.synthetic

pilot:
	PYTHONPATH=src python -m gaugeeeg.cli run --config configs/pilot.yaml --encoder bandpower
