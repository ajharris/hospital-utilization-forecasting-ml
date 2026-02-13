from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_raw: Path
    data_processed: Path
    models: Path
    reports: Path


@dataclass(frozen=True)
class Settings:
    random_seed: int = 42
    time_col: str = "timestamp"
    target_col: str = "y"


def get_project_root() -> Path:
    """Resolve project root from this file location."""
    return Path(__file__).resolve().parents[1]


def get_paths() -> ProjectPaths:
    """Return standard project paths without hardcoding absolute locations."""
    root = get_project_root()
    return ProjectPaths(
        root=root,
        data_raw=root / "data" / "raw",
        data_processed=root / "data" / "processed",
        models=root / "models",
        reports=root / "reports",
    )
