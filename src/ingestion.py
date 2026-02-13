from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.config import Settings, get_paths
from src.logging_config import configure_logging

logger = logging.getLogger(__name__)


def _get_provider(provider_name: str) -> Any:
    """Return a publicdata_ca provider instance based on a config string."""
    provider_name = provider_name.lower()
    if provider_name == "statcan":
        from publicdata_ca.providers import StatCanProvider

        return StatCanProvider()
    if provider_name == "open_canada":
        from publicdata_ca.providers import OpenCanadaProvider

        return OpenCanadaProvider()
    raise ValueError(f"Unsupported provider: {provider_name}")


def _make_dataset_ref(settings: Settings) -> Any:
    """Create a DatasetRef for publicdata_ca."""
    try:
        from publicdata_ca.provider import DatasetRef
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "publicdata_ca is required for ingestion. Install with `pip install publicdata-ca`."
        ) from exc

    return DatasetRef(
        provider=settings.dataset_provider,
        id=settings.dataset_id,
        params=settings.dataset_params or {},
    )


def _load_dataset_file(path: Path) -> pd.DataFrame:
    """Load a dataset file into a DataFrame based on file extension."""
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def _validate_dataset(df: pd.DataFrame) -> None:
    """Validate that dataset has basic expected shape."""
    if df.empty:
        raise ValueError("Dataset is empty.")
    if len(df.columns) == 0:
        raise ValueError("Dataset has no columns.")


def _write_metadata(
    df: pd.DataFrame,
    metadata_path: Path,
    settings: Settings,
    source_info: Dict[str, Any],
) -> None:
    """Write dataset metadata to disk as JSON."""
    date_range: Dict[str, str] | None = None
    if settings.time_col in df.columns:
        series = pd.to_datetime(df[settings.time_col], errors="coerce")
        if series.notna().any():
            date_range = {
                "min": series.min().isoformat(),
                "max": series.max().isoformat(),
            }

    metadata = {
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "date_range": date_range,
        "source": source_info,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def ingest(settings: Settings | None = None) -> Path:
    """Ingest raw data via publicdata_ca and store it in parquet format."""
    settings = settings or Settings()
    paths = get_paths()
    paths.data_raw.mkdir(parents=True, exist_ok=True)

    provider = _get_provider(settings.dataset_provider)
    ref = _make_dataset_ref(settings)
    logger.info("Fetching dataset %s:%s", settings.dataset_provider, settings.dataset_id)
    result = provider.fetch(ref, str(paths.data_raw))
    files = result.get("files") if isinstance(result, dict) else None
    if not files:
        raise ValueError("No files returned from provider fetch().")

    dataset_path = Path(files[0])
    df = _load_dataset_file(dataset_path)
    _validate_dataset(df)

    out_path = paths.data_raw / settings.raw_filename
    df.to_parquet(out_path, index=False)
    logger.info("Wrote raw data to %s", out_path)

    metadata_path = paths.reports / settings.metadata_filename
    source_info = {
        "provider": settings.dataset_provider,
        "dataset_id": settings.dataset_id,
        "params": settings.dataset_params,
        "downloaded_files": [str(p) for p in files],
    }
    _write_metadata(df, metadata_path, settings=settings, source_info=source_info)
    logger.info("Wrote metadata to %s", metadata_path)
    return out_path


if __name__ == "__main__":
    configure_logging()
    ingest()
