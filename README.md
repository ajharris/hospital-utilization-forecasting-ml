# Hospital Utilization Forecasting Pipeline

A reproducible machine learning pipeline scaffold for forecasting hospital utilization using time-series and hierarchical modeling approaches.

## Overview

This repo is a reproducible ML pipeline for forecasting monthly hospital utilization using time-series data. It covers ingestion, validation, feature engineering, model training, and evaluation, plus interpretation-ready artifacts.

The system demonstrates:

- Reproducible data ingestion
- Time-aware feature engineering
- Rolling cross-validation
- Statistical model evaluation
- Containerized ML workflows

## Problem Framing

Forecast next-month hospital utilization (e.g., admissions or ICU occupancy) from historical monthly time-series at the hospital/province level. The modeling constraints are:

- Avoid temporal leakage (strict time ordering).
- Account for heterogeneity across hospitals/provinces.
- Evaluate with time-aware splits to mirror deployment.

## Dataset Source

The pipeline ingests a public Statistics Canada dataset via `publicdata_ca`:

- Provider: `statcan`
- Dataset ID: `18100004`
- Date range (current ingest): 1914-01 to 2025-12
- Key columns used: `REF_DATE` (time), `GEO` (hospital/province), `VALUE` (target)

See `reports/dataset_metadata.json` for the exact schema and download provenance.

## Architecture

```
data/
  raw/
  processed/

src/
  ingestion.py
  validation.py
  features.py
  train.py
  backtest.py
  evaluate.py

configs/
models/
reports/
```

Pipeline Flow:

1. Ingest data and store as parquet
1. Validate and clean dataset
1. Generate lag and date-derived features
1. Train baseline regression model
1. Evaluate with rolling time-series cross-validation
1. Prepare structure for pooled vs hierarchical models
1. Log metrics and diagnostics

## Modeling Approach

- Baseline: last-observation carry-forward (LOCF).
- Regression: ridge regression on lag/rolling features.
- Hierarchical: mixed-effects regression (optional; see `reports/hierarchical_comparison.md`).

## Evaluation Strategy

To simulate deployment conditions:

- Time-based 80/20 split for holdout evaluation.
- Expanding window backtesting (4 folds, 20% test window).
- Metrics: RMSE, MAE, MAPE, R2.

Random cross-validation is intentionally avoided to prevent temporal leakage.

## How To Run

Install dependencies:

```bash
make install
```

Run ingestion:

```bash
make ingest
```

Train model:

```bash
make train
```

Run backtesting:

```bash
make backtest
```

Run the full pipeline:

```bash
make pipeline
```

Generate interpretation reports:

```bash
PYTHONPATH=. python scripts/generate_interpretation_reports.py
```

Or via Docker:

```bash
docker build -t hospital-forecast .
docker run --rm -v "$PWD":/app hospital-forecast
```

If you omit the bind mount, artifacts are written inside the container and discarded on exit.

Pipeline outputs are written to:

- `data/` (raw + processed datasets)
- `models/` (trained model artifacts)
- `reports/` (metrics and metadata)

## Key Results

Holdout (20% test split):

- Baseline LOCF: RMSE 62.08, MAE 47.13
- Ridge regression: RMSE 39.29, MAE 27.36, MAPE 20.15, R2 0.226

Expanding window backtest (4 folds, 20% test window):

- Mean RMSE 20.42 (std 13.78)
- Mean MAE 14.08 (std 9.77)
- Mean MAPE 12.03 (std 6.16)
- Mean R2 0.252 (std 0.063)

See `reports/metrics.json` and `reports/backtest_metrics.json` for full details.

## Interpretation Artifacts

The `reports/` folder includes:

- `summary.md`: concise findings and limitations.
- `feature_importance.md`: top linear coefficients from the ridge model.
- `residual_diagnostics.md`: residual distribution summary and normality test.
- `error_breakdown.md`: error slices by province and hospital size.
- `hierarchical_comparison.md`: pooled vs hierarchical model comparison.

## Future Work

- Add drift detection
- Add automated retraining trigger
- Deploy inference API via FastAPI
- Add experiment tracking via MLflow
