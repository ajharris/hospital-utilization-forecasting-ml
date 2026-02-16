from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Settings
from src.logging_config import configure_logging

logger = logging.getLogger(__name__)


def _rolling_feature(
    series: pd.Series,
    group_keys: pd.Series,
    window: int,
    agg: str,
) -> pd.Series:
    shifted = series.groupby(group_keys).shift(1)
    rolled = shifted.groupby(group_keys).rolling(window=window, min_periods=window)
    if agg == "mean":
        values = rolled.mean()
    elif agg == "std":
        values = rolled.std()
    else:
        raise ValueError(f"Unsupported rolling aggregation: {agg}")
    return values.reset_index(level=0, drop=True)


def _pct_change_feature(series: pd.Series, group_keys: pd.Series) -> pd.Series:
    shifted = series.groupby(group_keys).shift(1)
    return shifted.groupby(group_keys).pct_change()


def build_features(
    df: pd.DataFrame,
    settings: Settings | None = None,
    include_rolling_std: bool = True,
    include_province_agg: bool = False,
) -> pd.DataFrame:
    """Create time-series features.

    TODO: Add domain-specific features and holiday effects.
    """
    settings = settings or Settings()
    df = df.copy()
    time_col = settings.time_col
    target_col = settings.target_col

    hospital_col = settings.hospital_col
    province_col = settings.province_col

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([hospital_col, time_col]).reset_index(drop=True)

    feature_cols: list[str] = []

    df["day_of_week"] = df[time_col].dt.dayofweek
    df["month"] = df[time_col].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    feature_cols.extend(["day_of_week", "month", "month_sin", "month_cos"])

    for lag in (1, 3, 6):
        col = f"lag_{lag}"
        df[col] = df.groupby(hospital_col)[target_col].shift(lag)
        feature_cols.append(col)

    for window in (3, 6, 12):
        mean_col = f"roll_mean_{window}"
        df[mean_col] = _rolling_feature(
            df[target_col],
            df[hospital_col],
            window,
            "mean",
        )
        feature_cols.append(mean_col)
        if include_rolling_std:
            std_col = f"roll_std_{window}"
            df[std_col] = _rolling_feature(
                df[target_col],
                df[hospital_col],
                window,
                "std",
            )
            feature_cols.append(std_col)

    df["pct_change_1"] = _pct_change_feature(df[target_col], df[hospital_col])
    feature_cols.append("pct_change_1")

    if include_province_agg and province_col in df.columns:
        province_series = (
            df.groupby([province_col, time_col])[target_col]
            .mean()
            .reset_index()
            .sort_values([province_col, time_col])
        )
        province_series["prov_mean_lag_1"] = province_series.groupby(province_col)[
            target_col
        ].shift(1)
        feature_cols.append("prov_mean_lag_1")
        for window in (3, 6, 12):
            mean_col = f"prov_mean_roll_{window}"
            province_series[mean_col] = _rolling_feature(
                province_series[target_col],
                province_series[province_col],
                window,
                "mean",
            )
            feature_cols.append(mean_col)
            if include_rolling_std:
                std_col = f"prov_mean_roll_std_{window}"
                province_series[std_col] = _rolling_feature(
                    province_series[target_col],
                    province_series[province_col],
                    window,
                    "std",
                )
                feature_cols.append(std_col)
        df = df.merge(
            province_series.drop(columns=[target_col]),
            on=[province_col, time_col],
            how="left",
        )

    base_cols = [time_col, target_col, hospital_col]
    if province_col not in base_cols:
        base_cols.append(province_col)
    df = df[base_cols + feature_cols]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    return df


def run_feature_engineering(settings: Settings | None = None) -> Path:
    """Load validated data, build features, and write processed parquet."""
    settings = settings or Settings()
    # import get_paths at runtime so tests can monkeypatch src.config.get_paths
    from src.config import get_paths

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
