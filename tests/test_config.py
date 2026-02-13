"""Unit tests for src.config module."""

from pathlib import Path

import pytest

from src.config import ProjectPaths, Settings, get_paths, get_project_root


class TestProjectPaths:
    """Tests for ProjectPaths dataclass."""

    def test_project_paths_creation(self) -> None:
        """Test ProjectPaths can be instantiated."""
        root = Path("/test")
        paths = ProjectPaths(
            root=root,
            data_raw=root / "data" / "raw",
            data_processed=root / "data" / "processed",
            models=root / "models",
            reports=root / "reports",
        )
        assert paths.root == root
        assert paths.data_raw == root / "data" / "raw"

    def test_project_paths_immutable(self) -> None:
        """Test ProjectPaths is immutable (frozen)."""
        root = Path("/test")
        paths = ProjectPaths(
            root=root,
            data_raw=root / "data" / "raw",
            data_processed=root / "data" / "processed",
            models=root / "models",
            reports=root / "reports",
        )
        with pytest.raises(AttributeError):
            paths.root = Path("/other")


class TestSettings:
    """Tests for Settings dataclass."""

    def test_settings_defaults(self) -> None:
        """Test Settings has correct defaults."""
        settings = Settings()
        assert settings.random_seed == 42
        assert settings.time_col == "timestamp"
        assert settings.target_col == "y"

    def test_settings_custom_values(self) -> None:
        """Test Settings can be customized."""
        settings = Settings(random_seed=123, time_col="date", target_col="target")
        assert settings.random_seed == 123
        assert settings.time_col == "date"
        assert settings.target_col == "target"

    def test_settings_immutable(self) -> None:
        """Test Settings is immutable (frozen)."""
        settings = Settings()
        with pytest.raises(AttributeError):
            settings.random_seed = 100


class TestGetProjectRoot:
    """Tests for get_project_root function."""

    def test_get_project_root_returns_path(self) -> None:
        """Test get_project_root returns a Path object."""
        root = get_project_root()
        assert isinstance(root, Path)

    def test_get_project_root_exists(self) -> None:
        """Test returned project root exists."""
        root = get_project_root()
        assert root.exists()

    def test_get_project_root_has_src(self) -> None:
        """Test project root contains src directory."""
        root = get_project_root()
        assert (root / "src").exists()


class TestGetPaths:
    """Tests for get_paths function."""

    def test_get_paths_returns_project_paths(self) -> None:
        """Test get_paths returns ProjectPaths object."""
        paths = get_paths()
        assert isinstance(paths, ProjectPaths)

    def test_get_paths_all_attributes_are_paths(self) -> None:
        """Test all ProjectPaths attributes are Path objects."""
        paths = get_paths()
        assert isinstance(paths.root, Path)
        assert isinstance(paths.data_raw, Path)
        assert isinstance(paths.data_processed, Path)
        assert isinstance(paths.models, Path)
        assert isinstance(paths.reports, Path)

    def test_get_paths_structure(self) -> None:
        """Test get_paths creates correct directory structure."""
        paths = get_paths()
        assert paths.data_raw.parent == paths.root / "data"
        assert paths.data_processed.parent == paths.root / "data"
        assert paths.models.parent == paths.root
        assert paths.reports.parent == paths.root
