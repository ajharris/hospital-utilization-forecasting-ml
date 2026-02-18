.PHONY: help install ingest train backtest hierarchical test lint format features coverage validate

help:
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_.-]+:.*##/ {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	pip install -r requirements.txt

test: ## Run tests
	pytest tests/ -v

coverage: ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing

lint: ## Run linters
	ruff check src/ tests/
	black --check src/ tests/

format: ## Format code
	black src/ tests/
	ruff check --fix src/ tests/

ingest: ## Run ingestion and validation
	python -m src.ingestion
	python -m src.validation

validate: ## Run validation only
	python -m src.validation

train: ## Train model
	python -m src.train

backtest: ## Run backtest
	python -m src.backtest

hierarchical: ## Run hierarchical model
	python -m src.hierarchical

features: ## Build features
	python -m src.features
