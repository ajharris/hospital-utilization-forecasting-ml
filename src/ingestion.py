from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Settings, get_paths
from src.logging_config import configure_logging

logger = logging.getLogger(__name__)


def ingest(settings: Settings | None = None) -> Path:
    """Ingest raw data and store it in parquet format.

    TODO: Replace synthetic data generation with real ingestion logic
    (e.g., databases, APIs, S3, or file drops).
    """
    settings = settings or Settings()
    paths = get_paths()
    paths.data_raw.mkdir(parents=True, exist_ok=True)

    logger.info("Generating synthetic raw data (placeholder).")
    rng = np.random.default_rng(settings.random_seed)
    dates = pd.date_range("2020-01-01", periods=365, freq="D")
    values = rng.normal(loc=100.0, scale=10.0, size=len(dates))
    df = pd.DataFrame({settings.time_col: dates, settings.target_col: values})

    out_path = paths.data_raw / "raw.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Wrote raw data to %s", out_path)
    return out_path


if __name__ == "__main__":
    configure_logging()
    ingest()
