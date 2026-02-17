from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.config import Settings, get_paths
from src.experiments import append_experiment_record, build_run_record
from src.logging_config import configure_logging

logger = logging.getLogger(__name__)


def _feature_columns(df: pd.DataFrame, settings: Settings) -> list[str]:
    drop_cols = {
        settings.time_col,
        settings.target_col,
        settings.hospital_col,
        settings.province_col,
    }
    return [col for col in df.columns if col not in drop_cols]


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, y_true)
    value = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0
    return float(value)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mape = _mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2),
    }


def _sanitize_payload(value: object) -> object:
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: _sanitize_payload(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_sanitize_payload(item) for item in value]
    return value


def expanding_window_splits(
    df: pd.DataFrame,
    time_col: str,
    test_size: float,
    n_splits: int | None = None,
    min_train_size: int | None = None,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    n = len(df_sorted)
    test_n = int(n * test_size)
    if test_n <= 0:
        raise ValueError("test_size is too small for dataset length.")

    splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    train_end = min_train_size if min_train_size is not None else test_n
    while True:
        test_start = train_end
        test_end = test_start + test_n
        if test_end > n:
            break
        train = df_sorted.iloc[:train_end].copy()
        test = df_sorted.iloc[test_start:test_end].copy()
        if len(train) == 0 or len(test) == 0:
            break
        splits.append((train, test))
        if n_splits is not None and len(splits) >= n_splits:
            break
        train_end += test_n
    if not splits:
        raise ValueError("Not enough data to create expanding window splits.")
    return splits


def rolling_window_splits(
    df: pd.DataFrame,
    time_col: str,
    test_size: float,
    train_window: int,
    n_splits: int | None = None,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    if train_window <= 0:
        raise ValueError("train_window must be positive.")

    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    n = len(df_sorted)
    test_n = int(n * test_size)
    if test_n <= 0:
        raise ValueError("test_size is too small for dataset length.")

    splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    train_start = 0
    train_end = train_window
    while True:
        test_start = train_end
        test_end = test_start + test_n
        if test_end > n:
            break
        train = df_sorted.iloc[train_start:train_end].copy()
        test = df_sorted.iloc[test_start:test_end].copy()
        if len(train) == 0 or len(test) == 0:
            break
        splits.append((train, test))
        if n_splits is not None and len(splits) >= n_splits:
            break
        train_start += test_n
        train_end += test_n
    if not splits:
        raise ValueError("Not enough data to create rolling window splits.")
    return splits


def run_backtest(
    settings: Settings | None = None,
    *,
    window_type: str = "expanding",
    n_splits: int = 5,
    test_size: float = 0.2,
    min_train_size: int | None = None,
    rolling_train_size: int | None = None,
) -> Path:
    """Run time-series backtests and store metrics."""
    settings = settings or Settings()
    paths = get_paths()
    paths.reports.mkdir(parents=True, exist_ok=True)

    features_path = paths.data_processed / "features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Features data not found at {features_path}")

    df = pd.read_parquet(features_path)
    df[settings.time_col] = pd.to_datetime(df[settings.time_col])

    if window_type == "expanding":
        splits = expanding_window_splits(
            df,
            settings.time_col,
            test_size=test_size,
            n_splits=n_splits,
            min_train_size=min_train_size,
        )
    elif window_type == "rolling":
        if rolling_train_size is None:
            rolling_train_size = max(int(len(df) * 0.5), 1)
        splits = rolling_window_splits(
            df,
            settings.time_col,
            test_size=test_size,
            train_window=rolling_train_size,
            n_splits=n_splits,
        )
    else:
        raise ValueError("window_type must be 'expanding' or 'rolling'.")

    results: list[dict[str, object]] = []
    for idx, (train_df, test_df) in enumerate(splits):
        feature_cols = _feature_columns(train_df, settings)
        X_train = train_df[feature_cols]
        y_train = train_df[settings.target_col].to_numpy(dtype=float)
        X_test = test_df[feature_cols]
        y_test = test_df[settings.target_col].to_numpy(dtype=float)

        model = Ridge(alpha=1.0, random_state=settings.random_seed)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = _compute_metrics(y_test, preds)
        fold_result: dict[str, object] = {
            "fold": idx,
            "train_size": int(len(train_df)),
            "test_size": int(len(test_df)),
            "train_start": train_df[settings.time_col].min().isoformat(),
            "train_end": train_df[settings.time_col].max().isoformat(),
            "test_start": test_df[settings.time_col].min().isoformat(),
            "test_end": test_df[settings.time_col].max().isoformat(),
            "metrics": metrics,
        }
        results.append(fold_result)
        logger.info("Backtest fold %d metrics: %s", idx, metrics)

    metrics_df = pd.DataFrame([item["metrics"] for item in results])
    summary = metrics_df.agg(["mean", "std"]).to_dict()
    payload = {
        "window_type": window_type,
        "n_folds": len(results),
        "test_size": test_size,
        "min_train_size": min_train_size,
        "rolling_train_size": rolling_train_size,
        "folds": results,
        "summary": summary,
    }

    out_path = paths.reports / "backtest_metrics.json"
    out_path.write_text(json.dumps(_sanitize_payload(payload), indent=2, sort_keys=True))
    logger.info("Wrote backtest metrics to %s", out_path)

    run_record = build_run_record(
        model_type="ridge_regression_backtest",
        params={
            "window_type": window_type,
            "n_splits": n_splits,
            "test_size": test_size,
            "min_train_size": min_train_size,
            "rolling_train_size": rolling_train_size,
            "random_seed": settings.random_seed,
            "model_params": {
                "ridge_alpha": 1.0,
            },
        },
        metrics={
            "n_folds": len(results),
            "summary": summary,
        },
        settings=settings,
    )
    append_experiment_record(run_record)

    return out_path


if __name__ == "__main__":
    configure_logging()
    run_backtest()
