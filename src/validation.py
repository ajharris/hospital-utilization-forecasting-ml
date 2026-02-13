from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import Settings, get_paths
from src.logging_config import configure_logging

logger = logging.getLogger(__name__)


def validate_raw(input_path: Path, settings: Settings | None = None) -> pd.DataFrame:
    """Validate raw data and return a cleaned dataframe.

    TODO: Expand checks (schema validation, anomaly detection, unit tests).
    """
    settings = settings or Settings()
    df = pd.read_parquet(input_path)

    missing_cols = [c for c in [settings.time_col, settings.target_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.dropna(subset=[settings.time_col, settings.target_col]).copy()
    df[settings.time_col] = pd.to_datetime(df[settings.time_col])
    df = df.sort_values(settings.time_col).reset_index(drop=True)

    logger.info("Validated raw data with %d rows.", len(df))
    return df


def run_validation(settings: Settings | None = None) -> Path:
    """Load raw data, validate, and write processed parquet."""
    settings = settings or Settings()
    paths = get_paths()
    paths.data_processed.mkdir(parents=True, exist_ok=True)

    raw_path = paths.data_raw / "raw.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_path}")

    df = validate_raw(raw_path, settings=settings)
    out_path = paths.data_processed / "validated.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Wrote validated data to %s", out_path)
    return out_path


if __name__ == "__main__":
    configure_logging()
    run_validation()
