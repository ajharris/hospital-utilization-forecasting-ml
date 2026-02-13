"""Unit tests for src.ingestion module."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import ProjectPaths, Settings
from src.ingestion import _get_provider, ingest


class StubProvider:
    """Stub publicdata_ca provider for tests."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def fetch(self, ref: object, output_dir: str) -> dict[str, list[str]]:
        output_path = Path(output_dir) / "source.csv"
        self._df.to_csv(output_path, index=False)
        return {"files": [str(output_path)]}


@pytest.mark.unit
def test_ingest_writes_parquet_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ingestion writes parquet and metadata JSON."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=3, freq="D"),
            "y": [1.0, 2.0, 3.0],
        }
    )
    paths = ProjectPaths(
        root=tmp_path,
        data_raw=tmp_path / "data" / "raw",
        data_processed=tmp_path / "data" / "processed",
        models=tmp_path / "models",
        reports=tmp_path / "reports",
    )

    monkeypatch.setattr("src.ingestion.get_paths", lambda: paths)
    monkeypatch.setattr("src.ingestion._get_provider", lambda _: StubProvider(df))
    monkeypatch.setattr("src.ingestion._make_dataset_ref", lambda _: object())

    settings = Settings(
        dataset_provider="statcan",
        dataset_id="test_dataset",
        time_col="timestamp",
        target_col="y",
    )
    output_path = ingest(settings=settings)

    assert output_path.exists()
    reloaded = pd.read_parquet(output_path)
    assert len(reloaded) == 3
    assert list(reloaded.columns) == ["timestamp", "y"]

    metadata_path = paths.reports / settings.metadata_filename
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["row_count"] == 3
    assert metadata["columns"] == ["timestamp", "y"]
    assert metadata["date_range"]["min"].startswith("2021-01-01")
    assert metadata["date_range"]["max"].startswith("2021-01-03")


@pytest.mark.unit
def test_get_provider_unsupported() -> None:
    """Unsupported providers raise ValueError."""
    with pytest.raises(ValueError):
        _get_provider("unknown")


@pytest.mark.unit
def test_ingest_uses_local_cached_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ingestion can load from a local cached dataset file."""
    repo_root = Path(__file__).resolve().parents[1]
    raw_dir = repo_root / "data" / "raw"
    if not raw_dir.exists():
        pytest.skip("Cached dataset file not available.")

    candidates = [path for path in raw_dir.glob("*.csv") if "metadata" not in path.name.lower()]
    if not candidates:
        pytest.skip("No cached dataset CSV found in data/raw.")

    cached_file = candidates[0]

    paths = ProjectPaths(
        root=tmp_path,
        data_raw=tmp_path / "data" / "raw",
        data_processed=tmp_path / "data" / "processed",
        models=tmp_path / "models",
        reports=tmp_path / "reports",
    )

    class CachedProvider:
        def fetch(self, ref: object, output_dir: str) -> dict[str, list[str]]:
            return {"files": [str(cached_file)]}

    monkeypatch.setattr("src.ingestion.get_paths", lambda: paths)
    monkeypatch.setattr("src.ingestion._get_provider", lambda _: CachedProvider())
    monkeypatch.setattr("src.ingestion._make_dataset_ref", lambda _: object())

    settings = Settings(dataset_provider="statcan", dataset_id="18100004")
    output_path = ingest(settings=settings)

    assert output_path.exists()
    df = pd.read_parquet(output_path)
    assert not df.empty
