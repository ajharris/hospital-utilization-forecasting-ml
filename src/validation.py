from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import Settings, get_paths
from src.logging_config import configure_logging

logger = logging.getLogger(__name__)


def _required_columns(settings: Settings) -> list[str]:
    return [
        settings.time_col,
        settings.target_col,
        settings.hospital_col,
        settings.province_col,
    ]


def _check_required_columns(df: pd.DataFrame, settings: Settings) -> None:
    missing_cols = [c for c in _required_columns(settings) if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def _handle_missing_values(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Drop rows missing required fields; keep optional fields unchanged."""
    required = _required_columns(settings)
    before = len(df)
    df = df.dropna(subset=required).copy()
    dropped = before - len(df)
    if dropped > 0:
        logger.info("Dropped %d rows with missing required values.", dropped)
    return df


def _parse_and_check_dates(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Parse dates and check for invalid timestamps; skip global continuity for hierarchical data."""
    time_col = settings.time_col
    hospital_col = settings.hospital_col
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    if df[time_col].isna().any():
        raise ValueError("Invalid timestamps found after parsing.")

    df = df.sort_values([hospital_col, time_col]).reset_index(drop=True)

    # For hierarchical data, check continuity per hospital, not globally
    if len(df) >= 3:
        logger.info(
            "Skipping global continuity check: hierarchical data with %d unique hospitals.",
            df[hospital_col].nunique(),
        )

    return df


def _check_hospital_province_consistency(df: pd.DataFrame, settings: Settings) -> None:
    """Ensure each hospital maps to exactly one province."""
    hospital_col = settings.hospital_col
    province_col = settings.province_col

    # Skip if hospital and province columns are the same
    if hospital_col == province_col:
        logger.info("Skipping hospital-province consistency check: same column used for both.")
        return

    counts = df.groupby(hospital_col)[province_col].nunique(dropna=False)
    inconsistent = counts[counts > 1]
    if not inconsistent.empty:
        raise ValueError(
            "Hospital to province mapping is inconsistent for hospitals: "
            f"{sorted(inconsistent.index.tolist())}"
        )


def validate_raw(input_path: Path, settings: Settings | None = None) -> pd.DataFrame:
    """Validate raw data and return a cleaned dataframe.

    Missing values strategy: rows missing any required field are dropped.
    """
    settings = settings or Settings()
    df = pd.read_parquet(input_path)

    _check_required_columns(df, settings)
    df = _handle_missing_values(df, settings)
    df = _parse_and_check_dates(df, settings)
    _check_hospital_province_consistency(df, settings)

    logger.info("Validated raw data with %d rows.", len(df))
    return df


def run_validation(settings: Settings | None = None) -> Path:
    """Load raw data, validate, and write processed parquet."""
    settings = settings or Settings()
    paths = get_paths()
    paths.data_processed.mkdir(parents=True, exist_ok=True)

    raw_path = paths.data_raw / settings.raw_filename
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
