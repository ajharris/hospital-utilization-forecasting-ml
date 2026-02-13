from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.config import Settings, get_paths
from src.evaluate import evaluate_predictions
from src.logging_config import configure_logging
from src.utils import rolling_window_cv

logger = logging.getLogger(__name__)


def run_backtest(settings: Settings | None = None) -> Path:
    """Run rolling-window backtests and store metrics.

    TODO: Add hierarchical backtesting by group keys.
    """
    settings = settings or Settings()
    paths = get_paths()
    paths.reports.mkdir(parents=True, exist_ok=True)

    features_path = paths.data_processed / "features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Features data not found at {features_path}")

    df = pd.read_parquet(features_path)
    results: list[dict[str, float]] = []

    for idx, (train_df, test_df) in enumerate(
        rolling_window_cv(df, settings.time_col, n_splits=5, test_size=0.2)
    ):
        X_train = train_df.drop(columns=[settings.target_col])
        y_train = train_df[settings.target_col]
        X_test = test_df.drop(columns=[settings.target_col])
        y_test = test_df[settings.target_col]

        model = RandomForestRegressor(
            n_estimators=200,
            random_state=settings.random_seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate_predictions(y_test, preds)
        metrics["split"] = float(idx)
        results.append(metrics)
        logger.info("Backtest split %d metrics: %s", idx, metrics)

    metrics_df = pd.DataFrame(results)
    out_path = paths.reports / "backtest.parquet"
    metrics_df.to_parquet(out_path, index=False)
    logger.info("Wrote backtest report to %s", out_path)
    return out_path


if __name__ == "__main__":
    configure_logging()
    run_backtest()
