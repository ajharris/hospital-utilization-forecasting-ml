"""Unit tests for src.hierarchical module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ProjectPaths, Settings
from src.hierarchical import _subset_if_needed, run_hierarchical


def _make_features_df(settings: Settings, n_rows: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=n_rows, freq="D")
    hospitals = ["H1", "H2", "H3"]
    provinces = {"H1": "P1", "H2": "P1", "H3": "P2"}
    hospital_series = [hospitals[i % len(hospitals)] for i in range(n_rows)]
    province_series = [provinces[h] for h in hospital_series]

    df = pd.DataFrame(
        {
            settings.time_col: dates,
            settings.target_col: np.linspace(10.0, 25.0, n_rows),
            settings.hospital_col: hospital_series,
            settings.province_col: province_series,
            "feat_1": np.linspace(0.0, 1.0, n_rows),
            "feat_2": np.arange(n_rows, dtype=float) % 4,
        }
    )
    return df


def test_subset_if_needed_returns_note() -> None:
    settings = Settings()
    df = _make_features_df(settings, n_rows=50)
    subset, note = _subset_if_needed(df, settings.time_col, max_rows=10)
    assert len(subset) == 10
    assert note is not None


def test_run_hierarchical_writes_report(tmp_path, monkeypatch) -> None:
    settings = Settings(
        time_col="date",
        target_col="value",
        hospital_col="hospital",
        province_col="province",
    )
    df = _make_features_df(settings, n_rows=60)

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

    monkeypatch.setattr("src.hierarchical.get_paths", lambda: paths)

    report_path = run_hierarchical(settings=settings, test_size=0.2, max_rows=None)

    assert report_path.exists()
    report_text = report_path.read_text()
    assert "Hierarchical Model Comparison" in report_text
    assert "Pooled" in report_text
    assert "Hierarchical" in report_text
