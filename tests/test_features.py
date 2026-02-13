"""Unit tests for src.features module."""

from __future__ import annotations

import pandas as pd
import pandas.testing as pdt
import pytest

from src.config import Settings
from src.features import build_features


def _make_sample_df(settings: Settings) -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=15, freq="MS")
    df = pd.DataFrame(
        {
            settings.time_col: list(dates) * 2,
            settings.target_col: list(range(1, 16)) + list(range(101, 116)),
            settings.hospital_col: ["H1"] * 15 + ["H2"] * 15,
            settings.province_col: ["P1"] * 15 + ["P2"] * 15,
        }
    )
    return df


def test_build_features_no_future_leakage() -> None:
    settings = Settings(hospital_col="hospital", province_col="province")
    df = _make_sample_df(settings)

    features = build_features(
        df,
        settings=settings,
        include_rolling_std=True,
        include_province_agg=True,
    )

    df_future = df.copy()
    last_date = df_future[settings.time_col].max()
    mask = (df_future[settings.hospital_col] == "H1") & (df_future[settings.time_col] == last_date)
    df_future.loc[mask, settings.target_col] = 9999.0

    features_future = build_features(
        df_future,
        settings=settings,
        include_rolling_std=True,
        include_province_agg=True,
    )

    cutoff = last_date - pd.offsets.MonthBegin(1)
    features = features[features[settings.time_col] <= cutoff].sort_values(
        [settings.hospital_col, settings.time_col]
    )
    features_future = features_future[features_future[settings.time_col] <= cutoff].sort_values(
        [settings.hospital_col, settings.time_col]
    )

    compare_cols = [col for col in features.columns if col not in {settings.target_col}]
    pdt.assert_frame_equal(
        features[compare_cols].reset_index(drop=True),
        features_future[compare_cols].reset_index(drop=True),
    )


def test_build_features_expected_values() -> None:
    settings = Settings(hospital_col="hospital", province_col="province")
    df = _make_sample_df(settings)

    features = build_features(df, settings=settings)

    row = features[
        (features[settings.hospital_col] == "H1")
        & (features[settings.time_col] == pd.Timestamp("2022-01-01"))
    ].iloc[0]

    assert row["lag_1"] == 12
    assert row["lag_3"] == 10
    assert row["lag_6"] == 7
    assert row["roll_mean_3"] == (12 + 11 + 10) / 3
    assert row["roll_mean_6"] == sum(range(7, 13)) / 6
    assert row["roll_mean_12"] == sum(range(1, 13)) / 12
    assert row["pct_change_1"] == pytest.approx((12 - 11) / 11)
