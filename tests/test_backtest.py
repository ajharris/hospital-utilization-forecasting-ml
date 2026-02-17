"""Unit tests for src.backtest module."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    _sanitize_payload,
    expanding_window_splits,
    rolling_window_splits,
    run_backtest,
)
from src.config import ProjectPaths, Settings


def _make_features_df(settings: Settings, n_rows: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=n_rows, freq="D")
    df = pd.DataFrame(
        {
            settings.time_col: dates,
            settings.target_col: np.arange(n_rows, dtype=float),
            settings.hospital_col: ["H1"] * n_rows,
            settings.province_col: ["P1"] * n_rows,
            "feat_1": np.linspace(0.0, 1.0, n_rows),
            "feat_2": np.arange(n_rows, dtype=float) % 3,
        }
    )
    return df


def test_expanding_window_splits_no_leakage() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=20, freq="D"),
            "value": np.arange(20, dtype=float),
        }
    )

    splits = expanding_window_splits(df, "timestamp", test_size=0.2, n_splits=3)

    assert len(splits) == 3
    for train, test in splits:
        assert len(train) > 0
        assert len(test) == 4
        assert train["timestamp"].max() < test["timestamp"].min()


def test_rolling_window_splits_no_leakage() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=20, freq="D"),
            "value": np.arange(20, dtype=float),
        }
    )

    splits = rolling_window_splits(
        df,
        "timestamp",
        test_size=0.2,
        train_window=6,
        n_splits=3,
    )

    assert len(splits) == 3
    for train, test in splits:
        assert len(train) == 6
        assert len(test) == 4
        assert train["timestamp"].max() < test["timestamp"].min()


def test_run_backtest_persists_metrics(tmp_path, monkeypatch) -> None:
    settings = Settings(
        time_col="date",
        target_col="value",
        hospital_col="hospital",
        province_col="province",
    )
    df = _make_features_df(settings, n_rows=30)

    paths = ProjectPaths(
        root=tmp_path,
        data_raw=tmp_path / "data" / "raw",
        data_processed=tmp_path / "data" / "processed",
        models=tmp_path / "models",
        reports=tmp_path / "reports",
    )
    paths.data_processed.mkdir(parents=True, exist_ok=True)

    features_path = paths.data_processed / "features.parquet"
    df.to_parquet(features_path, index=False)

    monkeypatch.setattr("src.backtest.get_paths", lambda: paths)

    out_path = run_backtest(
        settings=settings,
        window_type="expanding",
        n_splits=2,
        test_size=0.2,
    )

    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["window_type"] == "expanding"
    assert payload["n_folds"] == 2
    assert len(payload["folds"]) == 2
    metrics = payload["folds"][0]["metrics"]
    assert set(metrics.keys()) == {"mae", "rmse", "mape", "r2"}


def test_sanitize_payload_converts_nan() -> None:
    payload = {"value": float("nan"), "nested": {"value": float("nan")}}
    sanitized = _sanitize_payload(payload)
    assert sanitized == {"value": None, "nested": {"value": None}}
