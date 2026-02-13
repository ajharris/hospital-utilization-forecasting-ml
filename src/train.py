from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.config import Settings, get_paths
from src.evaluate import evaluate_predictions
from src.logging_config import configure_logging
from src.utils import time_train_test_split

logger = logging.getLogger(__name__)


def train_model(settings: Settings | None = None) -> Path:
    """Train a baseline regression model on time-series features.

    TODO: Extend to hierarchical models with group-specific estimators.
    """
    settings = settings or Settings()
    paths = get_paths()
    paths.models.mkdir(parents=True, exist_ok=True)

    features_path = paths.data_processed / "features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Features data not found at {features_path}")

    df = pd.read_parquet(features_path)
    split = time_train_test_split(df, settings.time_col, test_size=0.2)

    X_train = split.train.drop(columns=[settings.target_col])
    y_train = split.train[settings.target_col]
    X_test = split.test.drop(columns=[settings.target_col])
    y_test = split.test[settings.target_col]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=settings.random_seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = evaluate_predictions(y_test, preds)
    logger.info("Holdout metrics: %s", metrics)

    model_path = paths.models / "baseline_model.joblib"
    joblib.dump(model, model_path)
    logger.info("Saved model to %s", model_path)
    return model_path


if __name__ == "__main__":
    configure_logging()
    train_model()
