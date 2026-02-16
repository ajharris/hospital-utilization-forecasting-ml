"""Unit tests for src.train module."""

from __future__ import annotations

import json

import joblib
import pandas as pd
import pytest

from src.config import ProjectPaths, Settings
from src.train import _baseline_locf, _feature_columns, train_models


def _make_features_df(settings: Settings) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=10, freq="MS")
    df = pd.DataFrame(
        {
            settings.time_col: list(dates) * 2,
            settings.target_col: list(range(1, 11)) + list(range(11, 21)),
            settings.hospital_col: ["H1"] * 10 + ["H2"] * 10,
            settings.province_col: ["P1"] * 10 + ["P2"] * 10,
        }
    )
    df["feat_1"] = df[settings.target_col] * 2
    df["feat_2"] = df[settings.target_col] % 3
    return df


def test_train_models_persists_outputs(tmp_path, monkeypatch) -> None:
    settings = Settings(
        time_col="date",
        target_col="value",
        hospital_col="hospital",
        province_col="province",
    )
    df = _make_features_df(settings)

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
    monkeypatch.setattr("src.config.get_paths", lambda: paths)

    output = train_models(settings=settings, test_size=0.2)

    assert output["baseline_model"].exists()
    assert output["regression_model"].exists()
    assert output["metrics_path"].exists()

    metrics = json.loads(output["metrics_path"].read_text())
    assert "baseline_locf" in metrics
    assert "ridge_regression" in metrics
    assert set(metrics["baseline_locf"].keys()) == {"mae", "rmse"}
    assert set(metrics["ridge_regression"].keys()) == {"mae", "rmse"}


def test_train_models_reproducible(tmp_path, monkeypatch) -> None:
    settings = Settings(
        time_col="date",
        target_col="value",
        hospital_col="hospital",
        province_col="province",
    )
    df = _make_features_df(settings)

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
    monkeypatch.setattr("src.config.get_paths", lambda: paths)

    first = train_models(settings=settings, test_size=0.2)
    first_metrics = json.loads(first["metrics_path"].read_text())

    second = train_models(settings=settings, test_size=0.2)
    second_metrics = json.loads(second["metrics_path"].read_text())

    assert first_metrics == second_metrics


def test_feature_columns_excludes_metadata_cols() -> None:
    """Test that _feature_columns excludes time, target, and matching columns."""
    settings = Settings(
        time_col="date",
        target_col="value",
        hospital_col="hospital",
        province_col="province",
    )
    df = pd.DataFrame(
        {
            "date": [1, 2],
            "value": [10, 20],
            "hospital": ["H1", "H2"],
            "province": ["P1", "P2"],
            "feat_1": [100, 200],
            "feat_2": [300, 400],
        }
    )

    cols = _feature_columns(df, settings)

    assert cols == ["feat_1", "feat_2"]
    assert "date" not in cols
    assert "value" not in cols
    assert "hospital" not in cols
    assert "province" not in cols


def test_baseline_locf_uses_last_values() -> None:
    """Test that baseline LOCF returns last observation per hospital."""
    settings = Settings(
        time_col="date",
        target_col="value",
        hospital_col="hospital",
        province_col="province",
    )

    train_df = pd.DataFrame(
        {
            "date": pd.date_range("2022-01-01", periods=4, freq="MS"),
            "value": [10, 20, 30, 40],
            "hospital": ["H1", "H1", "H2", "H2"],
            "province": ["P1", "P1", "P2", "P2"],
        }
    )

    test_df = pd.DataFrame(
        {
            "date": pd.date_range("2022-05-01", periods=2, freq="MS"),
            "value": [0, 0],
            "hospital": ["H1", "H2"],
            "province": ["P1", "P2"],
        }
    )

    preds, model_state = _baseline_locf(train_df, test_df, settings)

    # H1 last value is 20, H2 last value is 40
    assert preds[0] == 20.0
    assert preds[1] == 40.0
    assert isinstance(model_state, dict), f"model_state should be a dict, got {type(model_state)}"
    assert model_state["type"] == "locf"
    assert isinstance(
        model_state.get("last_values"), dict
    ), f"last_values should be a dict, got {type(model_state.get('last_values'))}"
    assert model_state["last_values"]["H1"] == 20.0
    assert model_state["last_values"]["H2"] == 40.0


def test_baseline_locf_uses_fallback_for_unseen_hospital() -> None:
    """Test that baseline LOCF uses fallback value for hospitals not in training."""
    settings = Settings(
        time_col="date",
        target_col="value",
        hospital_col="hospital",
        province_col="province",
    )

    train_df = pd.DataFrame(
        {
            "date": pd.date_range("2022-01-01", periods=2, freq="MS"),
            "value": [10, 20],
            "hospital": ["H1", "H1"],
            "province": ["P1", "P1"],
        }
    )

    test_df = pd.DataFrame(
        {
            "date": pd.date_range("2022-03-01", periods=2, freq="MS"),
            "value": [0, 0],
            "hospital": ["H1", "H3"],  # H3 not in train
            "province": ["P1", "P3"],
        }
    )

    preds, model_state = _baseline_locf(train_df, test_df, settings)

    assert preds[0] == 20.0  # H1's last value
    assert preds[1] == 20.0  # H3 not in train, uses fallback


def test_train_models_missing_features_raises_error(tmp_path, monkeypatch) -> None:
    """Test that train_models raises error when features.parquet is missing."""
    settings = Settings(
        time_col="date",
        target_col="value",
        hospital_col="hospital",
        province_col="province",
    )

    paths = ProjectPaths(
        root=tmp_path,
        data_raw=tmp_path / "data" / "raw",
        data_processed=tmp_path / "data" / "processed",
        models=tmp_path / "models",
        reports=tmp_path / "reports",
    )
    paths.data_processed.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("src.config.get_paths", lambda: paths)

    with pytest.raises(FileNotFoundError, match="Features data not found"):
        train_models(settings=settings)


def test_train_models_loads_saved_models(tmp_path, monkeypatch) -> None:
    """Test that saved model artifacts can be loaded and used for prediction."""
    settings = Settings(
        time_col="date",
        target_col="value",
        hospital_col="hospital",
        province_col="province",
    )
    df = _make_features_df(settings)

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
    monkeypatch.setattr("src.config.get_paths", lambda: paths)

    output = train_models(settings=settings, test_size=0.2)

    # Load and verify baseline model
    baseline_model = joblib.load(output["baseline_model"])
    assert "type" in baseline_model
    assert baseline_model["type"] == "locf"
    assert "last_values" in baseline_model
    assert "fallback" in baseline_model

    # Load and verify regression model
    regression_model = joblib.load(output["regression_model"])
    assert hasattr(regression_model, "predict")
    assert hasattr(regression_model, "coef_")


def test_train_models_predictions_in_reasonable_range(tmp_path, monkeypatch) -> None:
    """Test that model predictions are within reasonable bounds."""
    settings = Settings(
        time_col="date",
        target_col="value",
        hospital_col="hospital",
        province_col="province",
    )
    df = _make_features_df(settings)

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
    monkeypatch.setattr("src.config.get_paths", lambda: paths)

    output = train_models(settings=settings, test_size=0.2)

    metrics = json.loads(output["metrics_path"].read_text())

    # Verify metrics are positive numbers
    for model_name in ["baseline_locf", "ridge_regression"]:
        mae = metrics[model_name]["mae"]
        rmse = metrics[model_name]["rmse"]
        assert isinstance(mae, (int, float)) and mae >= 0
        assert isinstance(rmse, (int, float)) and rmse >= 0
        assert rmse >= mae  # RMSE should be >= MAE by definition
