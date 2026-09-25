.PHONY: install lint test smoke prepare split train evaluate tune report clean-runs

PY ?= python

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests
	mypy src/insectpose/contracts.py src/insectpose/registry.py src/insectpose/evaluation

test:
	pytest -q

smoke:
	pytest -q -m smoke

prepare:
	$(PY) -m insectpose.cli prepare $(ARGS)

split:
	$(PY) -m insectpose.cli split $(ARGS)

train:
	$(PY) -m insectpose.cli train $(ARGS)

evaluate:
	$(PY) -m insectpose.cli evaluate $(ARGS)

tune:
	$(PY) -m insectpose.cli tune $(ARGS)

report:
	$(PY) -m insectpose.cli report $(ARGS)

# Supprime uniquement les runs incomplets (sans manifeste), cf. CONVENTIONS.md 8.2
clean-runs:
	$(PY) -c "from insectpose.paths import ProjectPaths; from insectpose.utils.io import purge_incomplete_runs; print(purge_incomplete_runs(ProjectPaths.default().runs))"
