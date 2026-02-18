from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import Settings

logger = logging.getLogger(__name__)


def _hash_file(path: Path, *, algo: str = "sha256", chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.new(algo)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            hasher.update(chunk)
    return f"{algo}:{hasher.hexdigest()}"


def _sanitize_payload(value: Any) -> Any:
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, dict):
        return {key: _sanitize_payload(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_sanitize_payload(item) for item in value]
    return value


def dataset_fingerprint(settings: Settings | None = None) -> str:
    # Import at runtime so tests can monkeypatch src.config.get_paths
    from src.config import get_paths

    settings = settings or Settings()
    paths = get_paths()
    features_path = paths.data_processed / "features.parquet"
    if features_path.exists():
        return _hash_file(features_path)
    raw_path = paths.data_raw / settings.raw_filename
    if raw_path.exists():
        return _hash_file(raw_path)
    raise FileNotFoundError("No dataset file found to fingerprint.")


def git_commit_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    sha = result.stdout.strip()
    return sha or None


def append_experiment_record(record: dict[str, Any]) -> Path:
    # Import at runtime so tests can monkeypatch src.config.get_paths
    from src.config import get_paths

    paths = get_paths()
    paths.reports.mkdir(parents=True, exist_ok=True)
    out_path = paths.reports / "experiments.jsonl"
    payload = json.dumps(record, sort_keys=True)
    with out_path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")
    logger.info("Appended experiment record to %s", out_path)
    return out_path


def build_run_record(
    *,
    model_type: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    settings: Settings | None = None,
) -> dict[str, Any]:
    record_settings = settings or Settings()
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": dataset_fingerprint(record_settings),
        "model_type": model_type,
        "params": _sanitize_payload(params),
        "metrics": _sanitize_payload(metrics),
        "git_commit_sha": git_commit_sha(),
    }
    return record
