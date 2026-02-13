.PHONY: install ingest train backtest test lint format

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
	python -m src.features

train:
	python -m src.train

backtest:
	python -m src.backtest
