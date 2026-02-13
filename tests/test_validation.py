"""Unit tests for src.validation module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import ProjectPaths, Settings
from src.validation import run_validation, validate_raw


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


@pytest.mark.unit
def test_validation_missing_required_columns(tmp_path: Path) -> None:
    """Missing required columns raises."""
    settings = Settings()
    df = pd.DataFrame({settings.time_col: ["2021-01-01"], settings.target_col: [1.0]})
    input_path = tmp_path / "raw.parquet"
    _write_parquet(df, input_path)

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_raw(input_path, settings=settings)


@pytest.mark.unit
def test_validation_invalid_dates(tmp_path: Path) -> None:
    """Invalid date parsing raises."""
    settings = Settings()
    df = pd.DataFrame(
        {
            settings.time_col: ["2021-01-01", "not-a-date"],
            settings.target_col: [1.0, 2.0],
            settings.hospital_col: ["H1", "H1"],
            settings.province_col: ["P1", "P1"],
        }
    )
    input_path = tmp_path / "raw.parquet"
    _write_parquet(df, input_path)

    with pytest.raises(ValueError, match="Invalid timestamps"):
        validate_raw(input_path, settings=settings)


@pytest.mark.unit
def test_validation_continuity_skipped_for_hierarchical(tmp_path: Path) -> None:
    """Hierarchical data skips global continuity check."""
    settings = Settings()
    df = pd.DataFrame(
        {
            settings.time_col: ["2021-01-01", "2021-01-02", "2021-01-04"],
            settings.target_col: [1.0, 2.0, 3.0],
            settings.hospital_col: ["H1", "H1", "H1"],
            settings.province_col: ["P1", "P1", "P1"],
        }
    )
    input_path = tmp_path / "raw.parquet"
    _write_parquet(df, input_path)

    # Should not raise even with non-continuous timestamps (hierarchical data)
    result = validate_raw(input_path, settings=settings)
    assert len(result) == 3


@pytest.mark.unit
def test_validation_hospital_province_inconsistent(tmp_path: Path) -> None:
    """Hospital mapping to multiple provinces raises."""
    settings = Settings(
        hospital_col="hospital",
        province_col="province",
    )
    df = pd.DataFrame(
        {
            settings.time_col: ["2021-01-01", "2021-01-02", "2021-01-03"],
            settings.target_col: [1.0, 2.0, 3.0],
            settings.hospital_col: ["H1", "H1", "H1"],
            settings.province_col: ["P1", "P2", "P2"],
        }
    )
    input_path = tmp_path / "raw.parquet"
    _write_parquet(df, input_path)

    with pytest.raises(ValueError, match="inconsistent"):
        validate_raw(input_path, settings=settings)


@pytest.mark.unit
def test_run_validation_writes_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """run_validation writes cleaned parquet to data/processed."""
    settings = Settings()
    df = pd.DataFrame(
        {
            settings.time_col: pd.date_range("2021-01-01", periods=3, freq="D"),
            settings.target_col: [1.0, 2.0, 3.0],
            settings.hospital_col: ["H1", "H1", "H1"],
            settings.province_col: ["P1", "P1", "P1"],
        }
    )

    paths = ProjectPaths(
        root=tmp_path,
        data_raw=tmp_path / "data" / "raw",
        data_processed=tmp_path / "data" / "processed",
        models=tmp_path / "models",
        reports=tmp_path / "reports",
    )
    raw_path = paths.data_raw / settings.raw_filename
    _write_parquet(df, raw_path)
    monkeypatch.setattr("src.validation.get_paths", lambda: paths)

    output_path = run_validation(settings=settings)
    assert output_path.exists()
