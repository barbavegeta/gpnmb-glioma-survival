.PHONY: setup download cohort survival figures all test

setup:
	python -m pip install -r requirements.txt

download:
	PYTHONPATH=. python scripts/00_download.py

cohort:
	PYTHONPATH=. python scripts/01_build_cohort.py

survival:
	PYTHONPATH=. python scripts/02_survival.py

figures:
	PYTHONPATH=. python scripts/03_figures.py

all: download cohort survival figures

test:
	PYTHONPATH=. pytest -q
