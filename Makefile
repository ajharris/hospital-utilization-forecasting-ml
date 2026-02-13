.PHONY: install ingest train backtest test lint format features

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

coverage:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/
	black --check src/ tests/

format:
	black src/ tests/
	ruff check --fix src/ tests/

ingest:
	python -m src.ingestion
	python -m src.validation

validate:
	python -m src.validation

train:
	python -m src.train

backtest:
	python -m src.backtest

features:
	python -m src.features
