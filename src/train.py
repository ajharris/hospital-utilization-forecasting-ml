from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.config import Settings
from src.evaluate import evaluate_predictions
from src.logging_config import configure_logging
from src.utils import time_train_test_split

logger = logging.getLogger(__name__)


def _feature_columns(df: pd.DataFrame, settings: Settings) -> list[str]:
    drop_cols = {
        settings.time_col,
        settings.target_col,
        settings.hospital_col,
        settings.province_col,
    }
    return [col for col in df.columns if col not in drop_cols]


def _baseline_locf(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    settings: Settings,
) -> tuple[np.ndarray, dict[str, object]]:
    train_sorted = train_df.sort_values(settings.time_col)
    last_values = (
        train_sorted.groupby(settings.hospital_col)[settings.target_col].last().astype(float)
    )
    fallback = float(train_sorted[settings.target_col].iloc[-1])
    preds = test_df[settings.hospital_col].map(last_values).fillna(fallback).to_numpy(dtype=float)
    model_state = {
        "type": "locf",
        "last_values": last_values.to_dict(),
        "fallback": fallback,
    }
    return preds, model_state


def train_models(settings: Settings | None = None, test_size: float = 0.2) -> dict[str, Path]:
    """Train baseline and regression models on time-series features."""
    settings = settings or Settings()
    np.random.seed(settings.random_seed)

    # import get_paths at runtime so tests can monkeypatch src.config.get_paths
    from src.config import get_paths

    paths = get_paths()
    paths.models.mkdir(parents=True, exist_ok=True)
    paths.reports.mkdir(parents=True, exist_ok=True)

    features_path = paths.data_processed / "features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Features data not found at {features_path}")

    df = pd.read_parquet(features_path)
    df[settings.time_col] = pd.to_datetime(df[settings.time_col])

    split = time_train_test_split(df, settings.time_col, test_size=test_size)
    y_test = split.test[settings.target_col].to_numpy(dtype=float)

    baseline_preds, baseline_state = _baseline_locf(split.train, split.test, settings)
    baseline_metrics = evaluate_predictions(y_test, baseline_preds)
    logger.info("Baseline LOCF metrics: %s", baseline_metrics)

    feature_cols = _feature_columns(split.train, settings)
    X_train = split.train[feature_cols]
    y_train = split.train[settings.target_col]
    X_test = split.test[feature_cols]

    regression = Ridge(alpha=1.0)
    regression.fit(X_train, y_train)
    reg_preds = regression.predict(X_test)
    regression_metrics = evaluate_predictions(y_test, reg_preds)
    logger.info("Regression metrics: %s", regression_metrics)

    baseline_path = paths.models / "baseline_locf.joblib"
    regression_path = paths.models / "ridge_regression.joblib"
    joblib.dump(baseline_state, baseline_path)
    joblib.dump(regression, regression_path)
    logger.info("Saved baseline model to %s", baseline_path)
    logger.info("Saved regression model to %s", regression_path)

    metrics = {
        "baseline_locf": baseline_metrics,
        "ridge_regression": regression_metrics,
    }
    metrics_path = paths.reports / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    logger.info("Saved metrics to %s", metrics_path)

    return {
        "baseline_model": baseline_path,
        "regression_model": regression_path,
        "metrics_path": metrics_path,
    }


if __name__ == "__main__":
    configure_logging()
    train_models()
