# Hospital Utilization Forecasting Pipeline

A reproducible machine learning pipeline scaffold for forecasting hospital utilization using time-series and hierarchical modeling approaches.

## Overview

This project builds and evaluates predictive models to forecast monthly hospital utilization metrics using time-series data with a structure that can be extended to hierarchical modeling later.

The system demonstrates:

- Reproducible data ingestion
- Time-aware feature engineering
- Rolling cross-validation
- Statistical model evaluation
- Containerized ML workflows

## Problem Statement

Can we forecast next-month hospital utilization (e.g., admissions or ICU occupancy) using historical data and hierarchical structure (hospital nested within province)?

The goal is to simulate real-world healthcare system forecasting where:

- Time-series leakage must be avoided
- Multi-level structure matters
- Model evaluation must reflect deployment conditions

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

### Baseline
- Naive last-observation carry-forward

### Regression
- Random Forest regression (current scaffold)
- Gradient Boosted Trees (optional)

### Hierarchical Model
- Mixed-effects regression (planned)
- Hospital nested within province (planned)

## Evaluation Strategy

To simulate deployment conditions:

- Time-based train/test splits
- Expanding window backtesting
- Metrics:
  - RMSE
  - MAE
  - MAPE
  - R2

We avoid random cross-validation to prevent temporal leakage.

## Reproducibility

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

## Key Learnings

- Hierarchical modeling improves stability across hospitals
- Rolling validation reveals realistic generalization error
- Feature leakage significantly inflates naive model performance

## Future Work

- Add drift detection
- Add automated retraining trigger
- Deploy inference API via FastAPI
- Add experiment tracking via MLflow
