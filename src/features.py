from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import Settings, get_paths
from src.logging_config import configure_logging

logger = logging.getLogger(__name__)


def build_features(df: pd.DataFrame, settings: Settings | None = None) -> pd.DataFrame:
    """Create time-series features.

    TODO: Add domain-specific features and holiday effects.
    """
    settings = settings or Settings()
    df = df.copy()
    time_col = settings.time_col
    target_col = settings.target_col

    df["day_of_week"] = df[time_col].dt.dayofweek
    df["month"] = df[time_col].dt.month
    df["lag_1"] = df[target_col].shift(1)
    df["lag_7"] = df[target_col].shift(7)
    df = df.dropna().reset_index(drop=True)
    return df


def run_feature_engineering(settings: Settings | None = None) -> Path:
    """Load validated data, build features, and write processed parquet."""
    settings = settings or Settings()
    paths = get_paths()
    paths.data_processed.mkdir(parents=True, exist_ok=True)

    validated_path = paths.data_processed / "validated.parquet"
    if not validated_path.exists():
        raise FileNotFoundError(f"Validated data not found at {validated_path}")

    df = pd.read_parquet(validated_path)
    features = build_features(df, settings=settings)

    out_path = paths.data_processed / "features.parquet"
    features.to_parquet(out_path, index=False)
    logger.info("Wrote features to %s", out_path)
    return out_path


if __name__ == "__main__":
    configure_logging()
    run_feature_engineering()
