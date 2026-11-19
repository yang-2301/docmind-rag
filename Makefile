PY ?= python3
export PYTHONPATH := src

.PHONY: help setup ingest eval dev test corpus clean

help:
	@echo "make setup   install runtime dependencies"
	@echo "make corpus  regenerate the demo corpus + golden set"
	@echo "make ingest  build the index from data/corpus"
	@echo "make eval    run the retrieval evaluation"
	@echo "make dev     start the API + UI on :8000"
	@echo "make test    run the test suite"
	@echo "make clean   delete the local index"

setup:
	$(PY) -m pip install -r requirements.txt

corpus:
	$(PY) scripts/make_corpus.py

ingest:
	$(PY) -m docmind.ingest.build

eval:
	$(PY) -m docmind.eval.run_eval

dev:
	$(PY) -m uvicorn docmind.api.main:app --reload --port 8000

test:
	$(PY) -m pytest -q

clean:
	rm -rf data/index
